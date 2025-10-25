from __future__ import annotations
import torch
import torch.nn as nn
from .layers import TransformerBlock, CrossAttentionBlock


class NodeEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, nlayers: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(nlayers)])

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        # nodes: [B, N, d_model]
        h = nodes
        for blk in self.blocks:
            h = blk(h)
        return h


class DepotEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, nlayers: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList([CrossAttentionBlock(d_model, nhead) for _ in range(nlayers)])

    def forward(self, depot: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        # depot: [B, 1, d_model], nodes: [B, N, d_model]
        h = depot
        for blk in self.blocks:
            h = blk(h, nodes, nodes)
        return h