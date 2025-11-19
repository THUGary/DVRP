from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    仅编码 nodes 与 depot 的 Encoder。

    输入（张量）:
      - nodes:     [B, N, 5]   (x, y, t_arrival, c/demand, t_due) — 顺序按仓库现有用法即可
      - node_mask: [B, N]      True 表示屏蔽该节点（无效/不可选）
    - depot:     [B, 1, 2]   (x, y)

    输出:
      - H_nodes: [B, N, d]
      - H_depot: [B, 1, d]
      - mask:    [B, N] 与输入 node_mask 对齐（可直接向下游传）
    """

    def __init__(self, d_model: int = 128, nhead: int = 8, nlayers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.node_proj = nn.Sequential(
            nn.Linear(5, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, norm_first=True
        )
        self.node_stack = nn.TransformerEncoder(encoder_layer, num_layers=nlayers, norm=nn.LayerNorm(d_model))

    def forward(self, feats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes: torch.Tensor = feats["nodes"]            # [B, N, 5]
        node_mask: torch.Tensor = feats["node_mask"]    # [B, N] (bool)
        depot: torch.Tensor = feats["depot"]            # [B, 1, 2]

        B, N, _ = nodes.shape

        # 将 depot 视作额外一个“节点”，补齐 (x, y, 0, 0, 0)
        depot_feats = torch.zeros(B, 1, 5, device=nodes.device, dtype=nodes.dtype)
        depot_feats[..., :2] = depot
        # TODO： Depot 的 due_time 设为稍大于 max_time 的值（例如 200），避免数值过大导致不稳定
        depot_feats[..., 4:] = 200.0
        tokens = torch.cat([depot_feats, nodes], dim=1)  # [B, 1+N, 5]

        H_tokens_in = self.node_proj(tokens)
        token_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=node_mask.device),
            node_mask,
        ], dim=1)  # [B, 1+N]

        H_tokens = self.node_stack(H_tokens_in, src_key_padding_mask=token_mask)
        H_depot = H_tokens[:, :1, :]
        H_nodes = H_tokens[:, 1:, :]
        return H_nodes, H_depot, node_mask
