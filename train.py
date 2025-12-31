from models.transformer import MyFirstTransformer
from batch import IWSTLDataset
from tokenizer import RustBPETokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from datetime import datetime

def train_one_epoch(model, loss_fn, optimizer, train_dataloader, device):
    model.train()
    for batch_idx, (src_x, tgt_x, tgt_y, src_pad, tgt_pad) in enumerate(train_dataloader):
        # set inputs to correct deivce
        src_x, tgt_x, tgt_y, src_pad, tgt_pad = src_x.to(device), tgt_x.to(device), tgt_y.to(device), src_pad.to(device), tgt_pad.to(device)
        model_out = model(src_x, tgt_x, src_pad, tgt_pad)
        logits = model_out.reshape(-1, model_out.size(-1))
        targets = tgt_y.reshape(-1)
        loss = loss_fn(logits, targets)

        # backward pass - compute gradient of loss and backpropogate
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # update params based on gradient
        optimizer.step()

        # print progress for batch
        if batch_idx % 200 == 0:
            print(f"Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

def eval(model, loss_fn, test_dataloader, device):
    # set model to eval
    model.eval()
    n_batches = len(test_dataloader)
    test_loss = 0
    total_correct, total_count = 0, 0

    with torch.no_grad():
        for src_x, tgt_x, tgt_y, src_pad, tgt_pad in test_dataloader:
            src_x, tgt_x, tgt_y, src_pad, tgt_pad = src_x.to(device), tgt_x.to(device), tgt_y.to(device), src_pad.to(device), tgt_pad.to(device)
            # compute prediction and loss
            model_out = model(src_x, tgt_x, src_pad, tgt_pad)
            logits = model_out.reshape(-1, model_out.size(-1))
            targets = tgt_y.reshape(-1) 
            test_loss += loss_fn(logits, targets).item()
            
            # compute accuracy
            pred = model_out.argmax(dim=-1)
            mask = (tgt_y != pad_id)
            total_correct += (pred[mask] == tgt_y[mask]).sum().item()
            total_count += mask.sum().item()
    
    # calcualte average loss and jaccard score
    test_loss /= n_batches
    avg_score = total_correct / max(1, total_count)
    print(f"Test Error: \n Score: {(avg_score):>0.1f}%, Avg loss: {test_loss:>8f} \n")

@torch.no_grad()
def greedy_decode(model, src, src_pad_mask, bos_id, eos_id, max_len, pad_id=None):
    device = src.device
    B = src.size(0)
    tgt = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_pad_mask = (tgt == pad_id) if pad_id is not None else None
        logits = model(src, tgt, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
        next_id = logits[:, -1, :].argmax(dim=-1)  # (B,)

        # once finished, keep emitting EOS (or PAD) so sequence stays stable
        next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)

        tgt = torch.cat([tgt, next_id[:, None]], dim=1)
        finished |= (next_id == eos_id)

        if finished.all():
            break

    return tgt

def sample_predictions(model, tokenizer, sentence_samples, bos_id, pad_id, eos_id, device, context_limit=256):
    def pad_to(x, L):
            x = x[:L]
            pad_len = L - x.numel()
            return F.pad(x, (0, pad_len), value=pad_id)

    model.eval()
    # tokenize/batch sentences
    src_list, src_mask_list = [], []
    for text in sentence_samples:

        ids = tokenizer.encode(text)
        ids = torch.tensor(ids, dtype=torch.long)

        # ensure <= context_limit
        ids = ids[:context_limit]

        src = pad_to(ids, context_limit)

        src_pad_mask = (src == pad_id)

        src_list.append(src)
        src_mask_list.append(src_pad_mask)
            
    src = torch.stack(src_list).to(device)
    src_pad_mask = torch.stack(src_mask_list).to(device)

    pred_targets = greedy_decode(model, src, src_pad_mask, bos_id, eos_id, context_limit)
    for i in range(pred_targets.shape[0]):
        sentence_tokens = pred_targets[i].tolist()
        if eos_id in sentence_tokens:
            sentence_tokens = sentence_tokens[:sentence_tokens.index(eos_id) + 1]
        pred_sentence = tokenizer.decode(sentence_tokens)
        
        print("Actual sentence:", sentence_samples[i], "predicted sentence:", pred_sentence)

if __name__ == "__main__":
    # load our datasets
    print("loading datasets")
    train_set = IWSTLDataset(set="train")
    test_set = IWSTLDataset(set="test")

    # add to dataloaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    # load tokenizer to get vocab size
    print("loading tokenizer")
    tokenizer = RustBPETokenizer("tokenizer/data/vocab.json", "tokenizer/data/merges.txt")
    vocab_size = tokenizer.vocab_size()
    special_ids = tokenizer.special_ids()
    bos_id, pad_id, eos_id = special_ids["bos"], special_ids["pad"], special_ids["eos"]

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # do training loop
    print("starting training")
    model = MyFirstTransformer(vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=3e-4)
    loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    n_epochs = 1
    for _ in range(n_epochs):
        train_one_epoch(model, loss, optimizer, train_loader, device)
        eval(model, loss, test_loader, device)

    # now visual check w/ sentence samples
    sentence_samples = [
        "How are we doing today??",
        "I really hope this model is kind of working...",
        "If not I'll be really sad hehe",
    ]

    sample_predictions(model, tokenizer, sentence_samples, bos_id, pad_id, eos_id, device)

    # save model
    torch.save(model.state_dict(), f"data/weights/transformer_seq2seq_{datetime.now().isoformat()}.pth")



