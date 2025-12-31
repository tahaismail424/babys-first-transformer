use anyhow::Result;
use tokenizer::tokenizer::{Tokenizer};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <vocab.json> <merges.txt>", args[0]);
        std::process::exit(1);
    }

    let vocab_path = &args[1];
    let merges_path = &args[2];

    let tok = Tokenizer::from_files(vocab_path, merges_path)?;

    println!("Loaded tokenizer.");
    println!("Vocab size: {}", tok.vocab_size());
    println!("Merge rules loaded: {}", tok.rank.len());
    println!(
        "Special IDs: pad={}, bos={}, eos={}, unk={}",
        tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id
    );

    // Quick sanity: lookup a couple of tokens to make sure they exist
    for probe in ["_", "e", "<bos>", "<eos>"] {
        match tok.token2id.get(probe) {
            Some(id) => println!("token '{}' -> id {}", probe, id),
            None => println!("token '{}' not in vocab", probe),
        }
    }

    // Now lets encode a sentence and print the encoding!!
    let example_sentence = "Hey, can you encode this for me?";
    let tokens = tok.encode(example_sentence, true, true);

    println!("Example encoding of following sentence:");
    println!("{}", example_sentence);
    println!("{:#?}", tokens);
    println!("Example decoding:");
    let to_decode = tokens.unwrap_or(Vec::<u32>::new());
    println!("{}", tok.decode(&to_decode, true));
    
    Ok(())
}
