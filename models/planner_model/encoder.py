from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    仅编码 nodes 与 depot 的 Encoder，并提供 agents 的编码接口（迁移自 Decoder）。

    输入（张量）:
      - nodes:     [B, N, 5]   (x, y, t_arrival, c/demand, t_due) — 顺序按仓库现有用法即可
      - node_mask: [B, N]      True 表示屏蔽该节点（无效/不可选）
      - depot:     [B, 1, 3]   (x, y, t_ref)

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
        self.depot_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, norm_first=True
        )
        self.node_stack = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # 可选：对 depot 做轻量变换（保持维度）
        self.depot_norm = nn.LayerNorm(d_model)

        # === Agents 编码（从 Decoder 迁移过来） ===
        # Agent 编码块：Embedding + Self-Attn + FFN（Transformer 风格：残差 + Norm）
        self.agent_mlp = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.agent_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.agent_attn_norm = nn.LayerNorm(d_model)
        self.agent_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model),
        )
        self.agent_ffn_norm = nn.LayerNorm(d_model)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes: torch.Tensor = feats["nodes"]            # [B, N, 5]
        node_mask: torch.Tensor = feats["node_mask"]    # [B, N] (bool)
        depot: torch.Tensor = feats["depot"]            # [B, 1, 3]

        H_nodes_in = self.node_proj(nodes)                 # [B, N, d]
        # 将 node_mask 作为 key_padding_mask 传入，True=pad/屏蔽
        # 当某个样本的所有 nodes 都被屏蔽（全 True）时，TransformerEncoder 的注意力可能产生 NaN。
        # 为此做安全处理：对“全被屏蔽”的样本，直接输出 0 向量；其余样本正常通过 encoder。
        B, N, d = H_nodes_in.shape
        if N == 0:
            H_nodes = H_nodes_in  # 空序列，直接返回（形状 [B,0,d]）
        else:
            all_masked = node_mask.all(dim=1)  # [B]
            if all_masked.any():
                H_nodes = torch.zeros_like(H_nodes_in)
                # 仅对非全屏蔽样本做编码
                keep_idx = (~all_masked).nonzero(as_tuple=False).squeeze(1)
                if keep_idx.numel() > 0:
                    H_nodes_keep = self.node_stack(
                        H_nodes_in.index_select(0, keep_idx),
                        src_key_padding_mask=node_mask.index_select(0, keep_idx),
                    )
                    H_nodes.index_copy_(0, keep_idx, H_nodes_keep)
                # 全屏蔽样本保持 0 向量（安全，不含 NaN）
            else:
                H_nodes = self.node_stack(H_nodes_in, src_key_padding_mask=node_mask)

        H_depot = self.depot_proj(depot)                # [B, 1, d]
        H_depot = self.depot_norm(H_depot)

        return H_nodes, H_depot, node_mask

    def encode_agents(self, agents_tensor: torch.Tensor) -> torch.Tensor:
        """
        将 agents 状态编码为 [B, A, d] 表征。
        输入: agents_tensor [B, A, 4]，四维含义为 (x, y, s, t_agent)
        """
        if agents_tensor is None:
            raise ValueError("encode_agents requires agents_tensor with shape [B,A,4]")
        h = self.agent_mlp(agents_tensor)                        # [B, A, d]
        # Self-Attn (Post-Attn Norm)
        attn_out, _ = self.agent_self_attn(h, h, h, need_weights=False)
        h = self.agent_attn_norm(h + attn_out)                   # Residual 1
        ffn_out = self.agent_ffn(h)
        h = self.agent_ffn_norm(h + ffn_out)                     # Residual 2
        return h