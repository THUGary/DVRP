from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, act=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            act(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # Pre-LN -> Attention -> Pre-LN -> FFN  (i.e. LN, attn, LN, ffn)
        # 1) LayerNorm before attention
        x_ln = self.norm1(x)
        h, _ = self.attn(x_ln, x_ln, x_ln, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        x = x + self.drop(h)
        # 2) LayerNorm before FFN
        x_ln2 = self.norm2(x)
        h2 = self.ff(x_ln2)
        x = x + self.drop(h2)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, key_padding_mask=None, attn_mask=None):
        # Pre-LN -> Attention -> Pre-LN -> FFN
        q_ln = self.norm1(q)
        h, _ = self.attn(q_ln, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        x = q + self.drop(h)
        x_ln = self.norm2(x)
        h2 = self.ff(x_ln)
        x = x + self.drop(h2)
        return x