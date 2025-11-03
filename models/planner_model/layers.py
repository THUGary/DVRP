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
        # Self-attention
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.drop(h))
        # FFN
        h2 = self.ff(x)
        x = self.norm2(x + self.drop(h2))
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
        h, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(q + self.drop(h))
        h2 = self.ff(x)
        x = self.norm2(x + self.drop(h2))
        return x