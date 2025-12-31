from .layers import EncoderStack, DecoderStack
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyFirstTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, context_limit=256, n_layers=6, n_heads=4, dropout=0.1):
        assert d_model % n_heads == 0, "number of attention heads must be divisible by dimensionality of model embeddings"
        super(MyFirstTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_limit = context_limit


        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_limit, d_model)
        self.drop = nn.Dropout(dropout)

        self.encoder = EncoderStack(n_layers, d_model=d_model, n_heads=n_heads)
        self.decoder = DecoderStack(n_layers, d_model=d_model, n_heads=n_heads)

        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False) # logits
        self.lm_head.weight = self.tok_embed.weight  # weight tying

    def _add_positional(self, x_tok_ids):
        B, T = x_tok_ids.shape
        pos = torch.arange(T, device=x_tok_ids.device) # (T,)
        return self.pos_embed(pos)[None, :, :]         # (1, T, d_model)

    def forward(self, src, tgt_in, src_pad_mask=None, tgt_pad_mask=None):
        # src: (B,S), tgt_in: (B, T)
        src_x = self.tok_embed(src) + self._add_positional(src)
        tgt_x = self.tok_embed(tgt_in) + self._add_positional(tgt_in)

        src_x = self.drop(src_x)
        tgt_x = self.drop(tgt_x)

        memory = self.encoder(src_x, src_pad_mask=src_pad_mask)
        dec = self.decoder(tgt_x,
            memory,
            tgt_pad_mask=tgt_pad_mask,
            src_pad_mask=src_pad_mask,
        )
        out = self.final_ln(dec)
        out = self.lm_head(out)
        return out
