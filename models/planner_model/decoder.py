from __future__ import annotations
from typing import Tuple, Dict, Optional

import math
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    解码器：
      1) agents 通过 MLP 处理后做一次多头自注意力编码
      2) 与 depot 拼接得到 context，经 MLP 投回 d 维
      3) 以 context 为 Query，对 (depot + nodes) 为 KV 做一次交叉注意力
        4) 用单独的缩放点积打分头对 context 与 (depot+nodes) 计算分数，输出 logits
            - 返回顺序为 [depot, nodes...]，形状 [B, A, N+1]
            - 上层若 A==1，会 squeeze 到 [B, N+1]

    说明：
    - mask: node_mask [B, N]（True=屏蔽），会在打分时对 nodes 段 (索引 1..N) 加 -inf；depot 位 (索引 0) 不屏蔽
    """

    def __init__(self, d_model: int = 128, nhead: int = 8) -> None:
        super().__init__()
        self.d_model = d_model

        # 1) Agents MLP + 自注意力
        self.agent_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.agent_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.agent_norm = nn.LayerNorm(d_model)

        # 2) 与 depot 拼接后降回 d 维
        self.ctx_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.ctx_norm = nn.LayerNorm(d_model)

        # 3) Context <-KV(depot+nodes) 的交叉注意力（更新 context）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # 4) 单独的打分头（缩放点积）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        enc_nodes: torch.Tensor,      # [B, N, d]
        enc_depot: torch.Tensor,      # [B, 1, d]
        node_mask: torch.Tensor,      # [B, N] (bool)
        agents_tensor: torch.Tensor,  # [B, A, 4]
    ) -> torch.Tensor:
        B, N, d = enc_nodes.shape
        A = agents_tensor.size(1)

        # 1) Agents 编码
        agents_h = self.agent_mlp(agents_tensor)                 # [B, A, d]
        agents_h2, _ = self.agent_self_attn(agents_h, agents_h, agents_h, need_weights=False)
        agents_h = self.agent_norm(agents_h + agents_h2)         # 残差

        # 2) 与 depot 拼接后过 MLP -> context（每个 agent 一个 context）
        depot_expand = enc_depot.expand(-1, A, -1)               # [B, A, d]
        context = torch.cat([agents_h, depot_expand], dim=-1)    # [B, A, 2d]
        context = self.ctx_proj(context)                         # [B, A, d]
        context = self.ctx_norm(context)

        if torch.isnan(enc_nodes).any() or torch.isinf(enc_nodes).any():
            print(f"[ERROR] nodes contains NaN/Inf")

        # 3) Joint-Attn over [Context, Depot, Nodes]，仅更新 Context 段
        kv = torch.cat([enc_depot, enc_nodes], dim=1)            # [B, 1+N, d]
        combo = torch.cat([context, kv], dim=1)                  # [B, A+1+N, d]
        # padding mask：Context 段不屏蔽；Depot 不屏蔽；Nodes 按 node_mask 屏蔽
        mask_ctx = torch.zeros(B, A, dtype=torch.bool, device=node_mask.device)   # [B,A]
        mask_depot = torch.zeros(B, 1, dtype=torch.bool, device=node_mask.device) # [B,1]
        combo_mask = torch.cat([mask_ctx, mask_depot, node_mask], dim=1)          # [B, A+1+N]
        combo2, _ = self.cross_attn(combo, combo, combo, key_padding_mask=combo_mask, need_weights=False)
        ctx2 = combo2[:, :A, :]                                                  # [B, A, d]
        context = self.cross_norm(context + ctx2)
        # 数值诊断（可选）
        if torch.isnan(context).any() or torch.isinf(context).any():
            print(f"[ERROR][cross-norm] context contains NaN/Inf")

        # 4) 打分（未归一化 logits）：scores = (Q K^T) / sqrt(d)
        Q = self.q_proj(context)                                 # [B, A, d]
        K = self.k_proj(kv)                                      # [B, 1+N, d]
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.d_model)  # [B, A, 1+N]
        # 如果有 nan 或 inf，则打印 scores
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print(f"[ERROR] scores contains NaN/Inf")

        # 将 nodes 段按 mask 置为 -inf；depot 段（index 0）不屏蔽
        # scores[..., 0] 对应 depot；scores[..., 1:] 对应 nodes
        neg_inf = torch.finfo(scores.dtype).min
        scores_nodes = scores[..., 1:]                           # [B, A, N]
        scores_nodes = scores_nodes.masked_fill(node_mask.unsqueeze(1), neg_inf)
        # 拼回 [depot, nodes...] 顺序
        logits = torch.cat([scores[..., 0:1], scores_nodes], dim=-1)
        return logits