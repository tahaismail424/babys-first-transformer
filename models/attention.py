import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def _split_heads(self, x):
        # x: (B, T, d_model) -> (B, H, T, d_head)
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.transpose(1, 2)
    
    def _merge_heads(self, x):
        # x: (B, H, T, d_head) -> (B, T, d_model)
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return x
    
    def forward(self, x, kv=None, key_pad_mask=None, causal=False):
        """
        x: (B, Tq, d_model)
        kv: (B, Tk, d_model) or None
        key_pad_mask: (B, Tk) bool tensor, True where PAD (mask out)
        causal: bool, apply causal mask (only for decoder self-attn)
        """
        if kv is None:
            kv = x # self-attn

        B, Tq, _ = x.shape
        _, Tk, _ = kv.shape

        q = self._split_heads(self.Wq(x))   # (B, H, Tq, Dh)
        k = self._split_heads(self.Wk(kv))  # (B, H, Tk, Dh)
        v = self._split_heads(self.Wv(kv))  # (B, H, Tk, Dh)

        # attention logits: (B, H, Tq, Tk)
        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh_head)

        # key padding mask: mask keys (Tk dimension)
        if key_pad_mask is not None:
            # key_pad_mask: (B, Tk) -> (B, 1, 1, Tk)
            attn_logits = attn_logits.masked_fill(key_pad_mask[:, None, None, :], float("-inf"))
        
        # causal mask: prevent attending to future keys (only meaningful when Tq==Tk and kv is x)
        if causal:
            # shape (Tq, Tk)
            causal_mask = torch.triu(
                torch.ones((Tq, Tk), device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_logits = attn_logits.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        attn = F.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v # (B, H, Tq, Dh)
        out = self._merge_heads(out) # (B, Tq, d_model)

        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out



