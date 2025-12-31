from .attention import MultiHeadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=None, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.mlp = MLP(d_model=d_model, d_ff=d_ff, dropout=dropout)
    
    def forward(self, x, src_pad_mask=None):
        x = x + self.self_attn(self.ln1(x), kv=None, key_pad_mask=src_pad_mask, causal=False)
        x = x + self.mlp(self.ln2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=4, d_ff=None, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        if d_ff is None:
            d_ff = 4 * d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.mlp = MLP(d_model=d_model, d_ff=d_ff, dropout=dropout)
    
    def forward(self, x, memory, tgt_pad_mask=None, src_pad_mask=None):
        # masked self-attention
        x = x + self.self_attn(self.ln1(x), kv=None, key_pad_mask=tgt_pad_mask, causal=True)
        # cross attention: queries fromm decoder, keys/values from encoder memory
        x = x + self.cross_attn(self.ln2(x), kv=memory, key_pad_mask=src_pad_mask, causal=False)
        # MLP
        x = x + self.mlp(self.ln3(x))
        return x
    
class EncoderStack(nn.Module):
    def __init__(self, n_layers, d_model=256, n_heads=4, d_ff=None, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, src_pad_mask=None):
        for layer in self.layers:
            x = layer(x, src_pad_mask=src_pad_mask)
        return x

class DecoderStack(nn.Module):
    def __init__(self, n_layers, d_model=256, n_heads=4, d_ff=None, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, memory, tgt_pad_mask=None, src_pad_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_pad_mask=tgt_pad_mask, src_pad_mask=src_pad_mask)
        return x