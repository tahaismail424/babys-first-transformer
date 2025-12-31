from utils import RANDOM_STATE
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from typing import Literal
import pandas as pd
import numpy as np
from tokenizer import RustBPETokenizer

TrainingObjective = Literal["seq2seq", "de2en", "en2de"]
SetType = Literal["train", "test"]

class IWSTLDataset(Dataset):
    def __init__(
            self, 
            data_path: str = "data/raw/iwstl2017-de-en_full.parquet", 
            tokenizer_vocab_path: str = "tokenizer/data/vocab.json",
            tokenizer_merges_path: str = "tokenizer/data/merges.txt",
            objective: TrainingObjective = "seq2seq", 
            set: SetType = "train",
            context_limit=256,
            train_size=0.8, 
            random_state=RANDOM_STATE,
            ):
        # load dataset as df
        sentence_df = pd.read_parquet(data_path)

        # load tokenizer
        tokenizer = RustBPETokenizer(tokenizer_vocab_path, tokenizer_merges_path)
        pad_id = tokenizer.special_ids()["pad"]

        # train test split
        indices = np.arange(len(sentence_df))
        train_idx, test_idx =  train_test_split(indices, train_size=train_size, random_state=random_state)

        if set == "train":
            set_sentences = sentence_df.iloc[train_idx].reset_index(drop=True)
        elif set == "test":
            set_sentences = sentence_df.iloc[test_idx].reset_index(drop=True)
        else:
            raise ValueError("set value can only either be 'train' or 'test'")

        src_list, tgt_in_list, tgt_out_list = [], [], []
        src_mask_list, tgt_mask_list = [], []

        def pad_to(x, L):
            x = x[:L]
            pad_len = L - x.numel()
            return F.pad(x, (0, pad_len), value=pad_id)

        for _, row in set_sentences.iterrows():
            # pick English for simplicity
            text = row.translation["en"]

            ids = tokenizer.encode(text)
            ids = torch.tensor(ids, dtype=torch.long)

            # ensure <= context_limit
            ids = ids[:context_limit]

            src = pad_to(ids, context_limit)

            tgt_in = pad_to(src[:-1], context_limit)
            tgt_out = pad_to(src[1:], context_limit)

            src_pad_mask = (src == pad_id)
            tgt_pad_mask = (tgt_in == pad_id)

            src_list.append(src)
            tgt_in_list.append(tgt_in)
            tgt_out_list.append(tgt_out)
            src_mask_list.append(src_pad_mask)
            tgt_mask_list.append(tgt_pad_mask)
            
        self.src = torch.stack(src_list)
        self.tgt_in = torch.stack(tgt_in_list)
        self.tgt_out = torch.stack(tgt_out_list)
        self.src_pad_mask = torch.stack(src_mask_list)
        self.tgt_pad_mask = torch.stack(tgt_mask_list)

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, idx):
        return (self.src[idx],
                self.tgt_in[idx],
                self.tgt_out[idx],
                self.src_pad_mask[idx],
                self.tgt_pad_mask[idx])