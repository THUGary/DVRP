from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TransformerBlock


class ContextDecoder(nn.Module):
    """
    融合 agent 与 depot 后，生成 agent_context 并对 [nodes, depot] 产生 logits。
    """
    def __init__(self, d_model: int, nhead: int, nlayers: int = 1):
        super().__init__()
        self.fuse = nn.Linear(2 * d_model, d_model)
        self.context_blocks = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(nlayers)])
        # 使用点积 + 可学习缩放 作为简单 logits 头
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, agent_embed: torch.Tensor, depot_embed: torch.Tensor,
                H_nodes: torch.Tensor, H_depot: torch.Tensor) -> torch.Tensor:
        """
        agent_embed: [B, 1, d_model]
        depot_embed: [B, 1, d_model]
        H_nodes: [B, N, d_model]
        H_depot: [B, 1, d_model]
        return logits: [B, N+1]
        """
        B, N, D = H_nodes.shape
        # concat agent & depot -> fuse
        ctx = torch.cat([agent_embed, depot_embed], dim=-1)  # [B,1,2D]
        ctx = self.fuse(ctx)  # [B,1,D]
        for blk in self.context_blocks:
            ctx = blk(ctx)
        # 计算与候选的相似度 logits，候选 = [nodes, depot]
        cand = torch.cat([H_nodes, H_depot], dim=1)  # [B,N+1,D]
        # [B,1,D] x [B,D,N+1] -> [B,1,N+1]
        logits = torch.matmul(ctx, cand.transpose(1, 2)) * self.scale
        logits = logits.squeeze(1)  # [B,N+1]
        return logits