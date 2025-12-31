use anyhow::{anyhow, Context, Result};
use serde_json;
use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::cmp::Reverse;


/// Special tokens appended at end of vocab
pub const SPECIAL_TOKENS: [&str; 4] = ["<pad>", "<bos>", "<eos>", "<unk>"];

#[derive(Debug)]
pub struct Tokenizer {
    // token string -> token id
    pub token2id: HashMap<String, u32>,
    // token id -> token string (Vec index is id)
    pub id2token: Vec<String>,
    // (id_a, id_b) -> merge priority (0 is highest priority)
    pub rank: HashMap<(u32, u32), u32>,
    // IDs of special tokens
    pub pad_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
    pub unk_id: u32,
}

impl Tokenizer {
    /// Load vocab.json (token ->) and merges.txt
    /// append special tokens to vocab and builds rank table
    pub fn from_files(vocab_path: &str, merges_path: &str) -> Result<Self> {
        // -------------------------
        // Load vocab.json (token -> id)
        // -------------------------
        let vocab_file = File::open(vocab_path)
            .with_context(|| format!("Failed to open vocab.json as {vocab_path}"))?;
        let mut token2id: HashMap<String, u32> = serde_json::from_reader(vocab_file)
            .with_context(|| "Failed to parse vcoab.json as {token: id} map")?;

        if token2id.is_empty() {
            return Err(anyhow!("vocab.json is empty"));
        }

        // Find max ID to size id2token correctly
        let max_id: u32 = token2id
            .values()
            .copied()
            .max()
            .ok_or_else(|| anyhow!("Could not determine max id in vocab..json"))?;

        // Build id2token with length max_id+1, fill with placeholder.
        let mut id2token = vec![String::new(); (max_id as usize) + 1];

        // Fill id2token and make sure ids are unique
        for (tok, &id) in token2id.iter() {
            let idx = id as usize;
            if !id2token[idx].is_empty() {
                return Err(anyhow!(
                    "Duplicate id {id} in vocab.json (tokens '{}' and '{}')",
                    id2token[idx],
                    tok
                ));
            }
            id2token[idx] = tok.clone()
        }

        // Sanity: ensure no gaps with empty slots (optional, but helpful).
        // If you *do* have gaps, it's still workable, but better to know.
        let gaps = id2token.iter().filter(|s| s.is_empty()).count();
        if gaps > 0 {
            eprintln!(
                "Warning: vocab.json has {gaps} unused id slots (gaps). This is OK, but unusual."
            );
        }

        // -------------------------
        // Append special tokens at the end
        // -------------------------
        let mut next_id: u32 = id2token.len() as u32;

        let mut special_ids = [0u32; 4];
        for (i, &tok) in SPECIAL_TOKENS.iter().enumerate() {
            if let Some(&existing) = token2id.get(tok) {
                // aif already have token in hashmap, reuse index
                special_ids[i] = existing;
            } else {
                token2id.insert(tok.to_string(), next_id);
                id2token.push(tok.to_string());
                special_ids[i] = next_id;
                next_id += 1;
            }
        }

        let pad_id = special_ids[0];
        let bos_id = special_ids[1];
        let eos_id = special_ids[2];
        let unk_id = special_ids[3];

        // -------------------------
        // Load merges.txt and build rank table
        // merges.txt format:
        //   #version: bpe
        //   tokenA tokenB
        //   tokenC tokenD
        // ...
        // -------------------------
        let merges_file = File::open(merges_path)
            .with_context(|| format!("Failed to open merges.txt at: {merges_path}"))?;
        let reader = BufReader::new(merges_file);

        let mut rank: HashMap<(u32, u32), u32> = HashMap::new();
        let mut priority: u32 = 0;

        for line_result in reader.lines() {
            let line = line_result
                .with_context(|| "Failed reading merges.txt line")?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Expect exactly two whitespace-separated tokens.
            let mut parts = line.split_whitespace();
            let a = parts.next().ok_or_else(|| anyhow!("Bad merges.txt line: '{line}'"))?;
            let b = parts.next().ok_or_else(|| anyhow!("Bad merges.txt line: '{line}'"))?;

            if parts.next().is_some() {
                return Err(anyhow!(
                    "Bad merges.txt line (expected 2 tokens): '{line}'"
                ));
            }

            let id_a = *token2id.get(a).ok_or_else(|| {
                anyhow!("merges.txt references token not in vocab: '{}'", a)
            })?;
            let id_b = *token2id.get(b).ok_or_else(|| {
                anyhow!("merges.txt references tokens not in vocab: '{}'", b)
            })?;

            // Lower priority number = earlier merge = higher priority.
            // If duplicates occur, keep the first one.
            rank.entry((id_a, id_b)).or_insert(priority);
            priority += 1;

        }

        Ok(Self {
            token2id,
            id2token,
            rank,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id2token.len()
    }

     /// Encode a sentence by:
    /// - splitting on whitespace
    /// - prefixing '_' on each word
    /// - BPE-encoding each word-unit and concatenating
    ///
    /// add BOS/EOS at the call site depending on encoder/decoder usage.)
    pub fn encode(&self, sequence: &str, add_bos: bool, add_eos: bool) -> Result<Vec<u32>> {
        let mut out: Vec<u32> = Vec::new();
        // prepend bos token id
        if add_bos {out.push(self.bos_id)};

        for word in sequence.split_whitespace() {
            // Build "_word"
            let mut unit = String::with_capacity(word.len() + 1);
            unit.push('_');
            unit.push_str(word);

            let ids = self.encode_word_unit_bpe(&unit)?;
            out.extend(ids);
        }

        // append eos token id
        if add_eos{out.push(self.eos_id)};

        Ok(out)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        // create string to hold decoded output
        let mut s = String::new();

        for &id in ids {
            if skip_special && (id == self.pad_id || id == self.bos_id || id == self.eos_id) {
                continue;
            }

            let tok = self.id2token
                .get(id as usize)
                .map(|t| t.as_str())
                .unwrap_or("<unk>");

            s.push_str(tok);
        }

        // Convert boundary marker '_' -> ' ' and trim.
        // This assumes '_' is your artificial whitespace marker.
        let mut out = String::with_capacity(s.len());
        for ch in s.chars() {
            if ch == '_' {
                out.push(' ');
            } else {
                out.push(ch);
            }
        }

        out.trim_start().to_string()
    }

    fn encode_word_unit_baseline(&self, unit: &str) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::with_capacity(unit.chars().count());

        for ch in unit.chars() {
            let s = ch.to_string();
            match self.token2id.get(&s) {
                Some(id) => out.push(*id),
                None => out.push(self.unk_id),
            }
        }

        out
    }

    /// Full BPE: heap-based merging using rank table.
    fn encode_word_unit_bpe(&self, unit: &str) -> Result<Vec<u32>> {
        // initialize base representation
        let base_ids = self.encode_word_unit_baseline(unit);
        let n = base_ids.len();

        if n == 0 {
            return Ok(Vec::new());
        }

        // Build node list (index-based linked structure)
        let mut nodes: Vec<Node> = Vec::with_capacity(n);
        for i in 0..n {
            let node = Node{
                id: base_ids[i],
                prev: if i == 0 { None } else { Some(i - 1) },
                next: if i == n - 1 { None } else { Some(i + 1) },
                alive: true,
            };
            nodes.push(node)
        }
        let head: usize = 0;

        // initialize heap from initial nodes
        let mut heap: BinaryHeap<(Reverse<u32>, usize, usize, u32, u32)> = BinaryHeap::new();
        for i in 0..(n - 1) {
            self.push_candidate(&nodes, &mut heap, i, i + 1);
        }

        // merge until our heap is exhausted (lazy invalidation)
        while let Some((Reverse(_r), i, j, left_id_snap, right_id_snap)) = heap.pop() {
            // skip candidate if no logner valid
            if !nodes[i].alive || !nodes[j].alive { continue };
            if nodes[i].next != Some(j) || nodes[j].prev != Some(i) { continue };
            if nodes[i].id != left_id_snap || nodes[j].id != right_id_snap { continue };

            // build merged string from token strings of the pair
            let left_tok = self.id2token
                .get(nodes[i].id as usize)
                .ok_or_else(|| anyhow!("id2token missing for id {}", nodes[i].id))?;
            let right_tok = self.id2token
                .get(nodes[j].id as usize)
                .ok_or_else(|| anyhow!("id2token missing for id {}", nodes[j].id))?;

            let mut merged_str = String::with_capacity(left_tok.len() + right_tok.len());
            merged_str.push_str(left_tok);
            merged_str.push_str(right_tok);

            // merged token must exist for our pair to have rank - throw error if this not case
            let merged_id  = *self.token2id
                .get(&merged_str)
                .ok_or_else(|| {
                    anyhow!("Merged token '{}' not found in vocab?", merged_str)
                })?;

                // Perform merge i+j -> (reuse left node), delete j
                let right_next: Option<usize>;
                {
                    let (left_node, right_node) = get_two_mut(&mut nodes, i, j);

                    // reuse left as merged
                    left_node.id = merged_id;

                    // remove right
                    right_node.alive = false;

                    // relink
                    right_next = right_node.next;
                    left_node.next = right_next;
                }
                // update back-link of node after j (fi any)
                if let Some(k) = right_next {
                    nodes[k].prev = Some(i);
                }

                // Push new candidates around i
                let prev_i = nodes[i].prev;
                let next_i = nodes[i].next;

                if let Some(p) = prev_i {
                    self.push_candidate(&nodes, &mut heap, p, i);
                }
                if let Some(k) = next_i {
                    self.push_candidate(&nodes, &mut heap, i, k);
                }
        }

        // now read out alive node values to get token list
        let mut out: Vec<u32> = Vec::new();
        let mut cur: Option<usize> = Some(head);
        while let Some(idx) = cur {
            if nodes[idx].alive { 
                out.push(nodes[idx].id);
            }
            cur = nodes[idx].next;
        }

        Ok(out)
    }

    fn push_candidate(
        &self, 
        nodes: &[Node], 
        heap: &mut BinaryHeap<(Reverse<u32>, usize, usize, u32, u32)>, 
        i: usize, 
        j: usize
    ) {
        // do nothing if nodes are not alive
        if !nodes[i].alive || !nodes[j].alive { return; }
        // do nothing if i and j are not adjacent in our nodelist
        if nodes[i].next != Some(j) || nodes[j].prev != Some(i) { return; } 

        let a = nodes[i].id;
        let b = nodes[j].id;
        if let Some(&r) = self.rank.get(&(a, b)) {
            heap.push((Reverse(r), i, j, a , b));
        }
    }

}


/// Private node struct for index-based linked list.
#[derive(Debug, Clone)]
struct Node {
    id: u32,
    prev: Option<usize>,
    next: Option<usize>,
    alive: bool,
}

// private function to get two distinct mutable references of node Vec
fn get_two_mut<T>(v: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j, "get_two_mut requires distinct indices");
    if i < j {
        let (left, right) = v.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = v.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
}