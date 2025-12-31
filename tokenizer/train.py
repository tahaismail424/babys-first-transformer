#from ..utils import RANDOM_STATE
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from collections import defaultdict
from typing import List, Tuple
from heapq import heapify, heappop, heappush

TARGET_VOCAB_SIZE = 200_000
max_iterations = 1_000_000

def train_tokenizer(data_path: str = "data/raw/iwstl2017-de-en_full.parquet") -> List[Tuple[str, str]]:
    # load dataset as df
    sentence_df = pd.read_parquet(data_path)

    # split into train and test sets
    indices = np.arange(len(sentence_df))
    train_indices, _ = train_test_split(indices, train_size=.80, random_state=42)

    # get training sentences
    training_sentences = sentence_df.loc[train_indices].reset_index(drop=True)

    # now iterate through sentences, storing word representations and frequencies (get our alphabet as initial vocab as well)
    current_vocab = set()
    maxint = 0
    sym2int = {}
    int2sym = {}
    word_reps = {}
    word_freqs = {}
    for _, sentence in training_sentences.iterrows():
        de_sentence = sentence.translation['de']
        en_sentence = sentence.translation['en']
        de_words = de_sentence.split()
        en_words = en_sentence.split()

        # iterate through German words
        for word in de_words:
            word_rep = f"_{word}"
            if word_rep in word_reps:
                word_freqs[word_rep] += 1
            else:
                word_freqs[word_rep] = 1
                symbols = list(word_rep)
                current_vocab.update(symbols)
                repr = []
                for symbol in symbols:
                    if symbol in sym2int:
                        repr.append(sym2int[symbol])
                    else:
                        sym2int[symbol] = maxint
                        int2sym[maxint] = symbol
                        repr.append(maxint)
                        maxint += 1
                word_reps[word_rep] = repr
        # iterate through English words
        for word in en_words:
            word_rep = f"_{word}"
            if word_rep in word_reps:
                word_freqs[word_rep] += 1
            else:
                word_freqs[word_rep] = 1
                symbols = list(word_rep)
                current_vocab.update(symbols)
                repr = []
                for symbol in symbols:
                    if symbol in sym2int:
                        repr.append(sym2int[symbol])
                    else:
                        sym2int[symbol] = maxint
                        int2sym[maxint] = symbol
                        repr.append(maxint)
                        maxint += 1
                word_reps[word_rep] = repr

    iteration = 0
    current_vocab_size = len(current_vocab)
    merge_rules = []
    print("finished getting all word representations. getting all initail pairs")
    while current_vocab_size < TARGET_VOCAB_SIZE and iteration < max_iterations:
        if iteration == 0:
            pair_freqs = defaultdict(int)
            pair_words = defaultdict(lambda: defaultdict(int))
            for word, sym_list in word_reps.items():
                counts = pair_counts(sym_list)
                for pair, count in counts.items():
                    pair_freqs[pair] += count * word_freqs[word]
                    pair_words[pair][word] = count
               
            # organize pair frequencies into a heap
            pair_heap = [(-freq, pair) for pair, freq in pair_freqs.items()]
            heapify(pair_heap)

        # pop pairs off max heap until freq matches our dict
        merge_found = False
        while pair_heap:
            freq, pair = heappop(pair_heap)
            freq = -freq
            actual_freq = pair_freqs[pair]
            if freq == actual_freq and freq > 0:
                merge_found = True
                break
        if not merge_found:
            break
        merge_rules.append(pair)

        # also add to our symbol index
        symbol_rep = "".join([int2sym[i] for i in pair])
        # print our current merge rule
        print(pair, " = ", symbol_rep)
        current_vocab.add(symbol_rep)
        sym2int[symbol_rep] = maxint
        int2sym[maxint] = symbol_rep
        maxint += 1
        current_rep = sym2int[symbol_rep]

        # update our word reps to reflect new pair joining
        affected_words = list(pair_words[pair].keys())
        for word in affected_words:
            # update counts
            old_counts = pair_counts(word_reps[word])
            for p, k in old_counts.items():
                pair_freqs[p] -= k * word_freqs[word]
                pair_words[p][word] -= k
                if pair_words[p][word] == 0: del pair_words[p][word]
                if len(pair_words[p].keys()) == 0: del pair_words[p]

            # rewrite representation of word
            new_rep = apply_merge(word_reps[word], pair[0], pair[1], current_rep)
            word_reps[word] = new_rep

            # now add the new contributions of the word
            new_counts = pair_counts(new_rep)
            for p, k in new_counts.items():
                pair_freqs[p] += k * word_freqs[word]
                pair_words[p][word] += k

            # push all pairs in old/new counts for lazy heap update
            pairs_to_push = old_counts.keys() | new_counts.keys()
            for item in pairs_to_push:
                to_push = (-pair_freqs[item], item)
                heappush(pair_heap, to_push)
        
        # update our vocab size and iteration number
        current_vocab_size = len(current_vocab)
        if iteration == 0:
            print("finished initial merge - performing remainders")
        iteration += 1
    print("finished training")
    return merge_rules, sym2int, int2sym

def save_tokenizer(merge_rules, sym2int, int2sym, output_path="tokenizer/data"):
    # save sym2int as json dictionary
    with open(f"{output_path}/vocab.json", "w") as f:
        json.dump(sym2int, f)
    
    # save merges as text tokens
    with open(f"{output_path}/merges.txt", "w") as f:
        header = "#version: bpe\n"
        f.write(header)
        for rule in merge_rules:
            pair = f"{int2sym[rule[0]]} {int2sym[rule[1]]}\n"
            f.write(pair)

def pair_counts(sym_list):
    counts = defaultdict(int)
    for i in range(len(sym_list) - 1):
        pair = (sym_list[i], sym_list[i + 1])
        counts[pair] += 1
    return counts

def apply_merge(sym_list, p1, p2, pair_rep):
    new_sym = []
    i = 0
    while i < len(sym_list):
        if i < len(sym_list) - 1 and sym_list[i] == p1 and sym_list[i + 1] == p2:
            new_sym.append(pair_rep)
            i += 2
        else:
            new_sym.append(sym_list[i])
            i += 1
    return new_sym

if __name__ == "__main__":
    merge_rules, sym2int, int2sym = train_tokenizer()
    save_tokenizer(merge_rules, sym2int, int2sym)
    print('success!')
