from __future__ import annotations
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import warnings

from .layers import CrossAttentionBlock, TransformerBlock


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

        # 历史序列处理：将位置和节点嵌入融合后交给 TransformerBlock
        self.history_block = TransformerBlock(d_model, nhead, dim_ff=d_model * 4)
        self.history_pos_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.history_target_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.history_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.history_pe_projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # 跨 agent 交互：先拼接 agent 编码和历史摘要，再用 TransformerBlock 聚合
        self.agent_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.agent_block = TransformerBlock(d_model, nhead, dim_ff=d_model * 4)

        # context 构建：将 agent / depot / 历史摘要拼接压回 d 维
        self.context_projection = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.context_norm = nn.LayerNorm(d_model)

        # agent 编码：从原 Encoder 迁移而来
        self.agent_embed = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.agent_embed_block = TransformerBlock(d_model, nhead, dim_ff=d_model * 4)

        # 交叉注意力：使用 CrossAttentionBlock 完成 cross-attn + FFN
        self.cross_block = CrossAttentionBlock(d_model, nhead, dim_ff=d_model * 4)

        # 最后打分头（缩放点积）
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        enc_nodes: torch.Tensor,      # [B, N, d]
        enc_depot: torch.Tensor,      # [B, 1, d]
        node_mask: torch.Tensor,      # [B, N] (bool)
        agents_tensor: torch.Tensor,  # [B, A, 4]
        history_positions: Optional[torch.Tensor] = None,       # [B, A, T, 2] agent 轨迹
        history_target_coords: Optional[torch.Tensor] = None,   # [B, A, T, 2] 目标节点坐标
    ) -> torch.Tensor:
        batch_size, num_nodes, d_model = enc_nodes.shape
        num_agents = agents_tensor.size(1)

        enc_agents = self.encode_agents(agents_tensor)
        history_summary = self.encode_history(
            enc_nodes=enc_nodes,
            enc_depot=enc_depot,
            num_nodes=num_nodes,
            num_agents=num_agents,
            history_positions=history_positions,
            history_target_coords=history_target_coords,
        )

        # --- 2) 跨-agent 历史交互 ---
        agent_context = torch.cat([enc_agents, history_summary], dim=-1)
        agent_context = self.agent_fusion(agent_context)
        agent_context = self.agent_block(agent_context)
        history_summary = agent_context

        # --- 3) 构造每个 agent 的 context ---
        depot_expanded = enc_depot.expand(-1, num_agents, -1)
        context = torch.cat([enc_agents, depot_expanded, history_summary], dim=-1)
        context = self.context_projection(context)
        context = self.context_norm(context)

        if torch.isnan(enc_nodes).any() or torch.isinf(enc_nodes).any():
            print("[ERROR] nodes contains NaN/Inf")

        # --- 4) Cross-Attn：Q=context, K/V=concat(depot, nodes) ---
        kv = torch.cat([enc_depot, enc_nodes], dim=1)
        kv_mask = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=node_mask.device),
            node_mask,
        ], dim=1)
        context = self.cross_block(context, kv, kv, key_padding_mask=kv_mask)
        if torch.isnan(context).any() or torch.isinf(context).any():
            print("[ERROR][cross-block] context contains NaN/Inf")

        # --- 5) 缩放点积打分 ---
        query = self.query_projection(context)
        key = self.key_projection(kv)
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.d_model)
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            print("[ERROR] scores contains NaN/Inf")

        # 先做 tanh 压缩数值范围，再做 mask
        scores = 10.0 * torch.tanh(scores)

        neg_inf = torch.finfo(scores.dtype).min
        node_scores = scores[..., 1:]
        node_scores = node_scores.masked_fill(node_mask.unsqueeze(1), neg_inf)
        logits = torch.cat([scores[..., :1], node_scores], dim=-1)
        return logits

    def encode_agents(self, agents_tensor: torch.Tensor) -> torch.Tensor:
        """将 agents (x, y, s, t) -> [B, A, d]"""
        if agents_tensor is None:
            raise ValueError("encode_agents requires agents_tensor with shape [B,A,4]")
        if agents_tensor.dim() != 3 or agents_tensor.size(-1) != 4:
            raise ValueError(
                f"agents_tensor must be [B,A,4], got {tuple(agents_tensor.shape)}"
            )
        h = self.agent_embed(agents_tensor)
        h = self.agent_embed_block(h)
        return h

    def encode_history(
        self,
        *,
        enc_nodes: torch.Tensor,
        enc_depot: torch.Tensor,
        num_nodes: int,
        num_agents: int,
        history_positions: Optional[torch.Tensor],
        history_target_coords: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = enc_nodes.size(0)
        device = enc_nodes.device
        dtype = enc_nodes.dtype
        if num_agents == 0:
            return torch.zeros(batch_size, 0, self.d_model, device=device, dtype=dtype)

        has_pos = history_positions is not None and history_positions.numel() > 0
        has_target = history_target_coords is not None and history_target_coords.numel() > 0
        if has_pos ^ has_target:
            warnings.warn(
                "history_positions 与 history_target_coords 必须同时提供或同时为空；检测到仅出现其中之一，将视为都未提供。",
                UserWarning,
            )
            return torch.zeros(batch_size, num_agents, self.d_model, device=device, dtype=dtype)

        if not (has_pos and has_target):
            return torch.zeros(batch_size, num_agents, self.d_model, device=device, dtype=dtype)

        hist_pos = history_positions  # [B, A, T, 2]
        hist_target = history_target_coords  # [B, A, T, 2]
        pad_pos = (hist_pos[..., 0] < 0) | (hist_pos[..., 1] < 0)
        pad_target = (hist_target[..., 0] < 0) | (hist_target[..., 1] < 0)
        pad_mask = pad_pos | pad_target

        batch_agents, hist_len = batch_size * num_agents, hist_pos.size(2)
        fused_hist = torch.cat(
            [
                self.history_pos_encoder(hist_pos.view(batch_agents, hist_len, 2)),
                self.history_target_encoder(hist_target.view(batch_agents, hist_len, 2)),
            ],
            dim=-1,
        )
        fused_hist = self.history_fusion(fused_hist)
        pe = self._build_sinusoidal_pe(hist_len, self.d_model, fused_hist.device)
        pe_expanded = pe.unsqueeze(0).expand(batch_agents, hist_len, -1)
        fused_hist = torch.cat([fused_hist, pe_expanded], dim=-1)
        fused_hist = self.history_pe_projection(fused_hist)
        fused_hist = self.history_block(
            fused_hist,
            key_padding_mask=pad_mask.view(batch_agents, hist_len),
        )
        valid_lengths = (~pad_mask.view(batch_agents, hist_len)).sum(dim=1)
        last_valid = torch.clamp(valid_lengths - 1, min=0)
        gather_index = torch.arange(batch_agents, device=fused_hist.device)
        history_summary = fused_hist[gather_index, last_valid, :].view(batch_size, num_agents, self.d_model)
        return history_summary

    @staticmethod
    def _build_sinusoidal_pe(T: int, d_model: int, device: torch.device) -> torch.Tensor:
        """生成标准 Transformer 正弦位置编码 [T, d_model]。
        pe[pos, 2i]   = sin(pos / (10000^(2i/d)))
        pe[pos, 2i+1] = cos(pos / (10000^(2i/d)))
        """
        position = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(1)  # [T,1]
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d_model))  # [d/2]
        pe = torch.zeros(T, d_model, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe