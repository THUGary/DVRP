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

        # 历史序列自注意力（可融合坐标与节点嵌入）
        self.hist_self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.hist_norm = nn.LayerNorm(d_model)
        # 融合后再加 FFN（可选强化表达）
        self.hist_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model),
        )
        self.hist_ffn_norm = nn.LayerNorm(d_model)

        # 跨 agent 的历史交互（让每个 agent 感知他人近期意图）
        # 先将 enc_agents 与 hist_summary 融合到 d 维，再在 agent 维度做自注意力
        self.agent_ctx_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.agent_ctx_norm = nn.LayerNorm(d_model)
        self.agent_interact_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.agent_interact_norm = nn.LayerNorm(d_model)
        self.agent_interact_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model),
        )
        self.agent_interact_ffn_norm = nn.LayerNorm(d_model)

        # 历史位置投影 (x,y) -> d
        self.hist_pos_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        # 位置 + 节点嵌入 融合线性（拼接 2d -> d）
        self.hist_fuse_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # 2) 与 depot 拼接后降回 d 维
        self.ctx_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.ctx_norm = nn.LayerNorm(d_model)

        # Cross-Attn (Q=context, K/V=kv) + FFN (Transformer 风格)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 4, d_model),
        )
        self.cross_ffn_norm = nn.LayerNorm(d_model)

        # 4) 单独的打分头（缩放点积）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        enc_nodes: torch.Tensor,      # [B, N, d]
        enc_depot: torch.Tensor,      # [B, 1, d]
        node_mask: torch.Tensor,      # [B, N] (bool)
        enc_agents: torch.Tensor,     # [B, A, d]
        history_indices: Optional[torch.Tensor] = None,  # [B, A, T] (每个值为 0..N-1；若为 -1 表示 padding)
        history_positions: Optional[torch.Tensor] = None, # [B, A, T, 2] (x,y)，若提供则优先使用
    ) -> torch.Tensor:
        B, N, d = enc_nodes.shape
        A = enc_agents.size(1)

        # 1) 历史编码（融合坐标 + 目标节点）
        if (history_positions is not None and history_positions.numel() > 0) and \
           (history_indices is not None and history_indices.numel() > 0):
            hp = history_positions  # [B,A,T,2]
            hi = history_indices    # [B,A,T]
            pad_pos = (hp[..., 0] < 0) | (hp[..., 1] < 0)
            pad_idx = hi < 0
            pad = pad_pos | pad_idx  # 综合 padding
            BA, T = B * A, hp.size(2)
            # 位置嵌入
            pos_emb = self.hist_pos_proj(hp.view(BA, T, 2))      # [BA,T,d]
            # 节点嵌入（索引 0=depot，1..N=nodes；-1 padding）
            idx_clamped = hi.clamp(min=0, max=N)                 # 允许取 N 表示越界暂不使用
            # 构造包含 depot + nodes 的全集嵌入以便索引：先拼接 depot 与 nodes
            full_nodes = torch.cat([enc_depot, enc_nodes], dim=1)  # [B,1+N,d]
            b_idx = torch.arange(B, device=enc_nodes.device).view(B,1,1).expand(B, A, T)
            # 防止越界：将 >N 的置为 0
            idx_full = idx_clamped.clamp(0, N)  # depot=0, nodes=1..N
            node_emb = full_nodes[b_idx, idx_full, :]             # [B,A,T,d]
            node_emb = node_emb.view(BA, T, d)
            # 融合（拼接 2d -> d）
            fuse = torch.cat([pos_emb, node_emb], dim=-1)         # [BA,T,2d]
            fuse = self.hist_fuse_proj(fuse)                      # [BA,T,d]
            pe = self._build_sinusoidal_pe(T, d, fuse.device)
            fuse = fuse + pe.unsqueeze(0)
            hist_out, _ = self.hist_self_attn(fuse, fuse, fuse, key_padding_mask=pad.view(BA, T), need_weights=False)
            lengths = (~pad.view(BA, T)).sum(dim=1)
            last_idx = torch.clamp(lengths - 1, min=0)
            arange_ba = torch.arange(BA, device=fuse.device)
            hist_summary = hist_out[arange_ba, last_idx, :].view(B, A, d)
            # FFN refine
            hist_summary = self.hist_norm(hist_summary)
            hist_summary = self.hist_ffn_norm(hist_summary + self.hist_ffn(hist_summary))
        elif history_positions is not None and history_positions.numel() > 0:
            hp = history_positions
            pad2d = (hp[..., 0] < 0) | (hp[..., 1] < 0)
            BA, T = B * A, hp.size(2)
            hp2d = hp.view(BA, T, 2)
            emb = self.hist_pos_proj(hp2d)
            pe = self._build_sinusoidal_pe(T, d, emb.device)
            emb = emb + pe.unsqueeze(0)
            hist_out, _ = self.hist_self_attn(emb, emb, emb, key_padding_mask=pad2d.view(BA, T), need_weights=False)
            lengths = (~pad2d.view(BA, T)).sum(dim=1)
            last_idx = torch.clamp(lengths - 1, min=0)
            arange_ba = torch.arange(BA, device=emb.device)
            hist_summary = hist_out[arange_ba, last_idx, :].view(B, A, d)
            hist_summary = self.hist_norm(hist_summary)
        elif history_indices is not None and history_indices.numel() > 0:
            hist_pad = history_indices < 0
            idx = history_indices.clamp(min=0, max=N-1)
            b_idx = torch.arange(B, device=enc_nodes.device).view(B,1,1).expand_as(idx)
            hist = enc_nodes[b_idx, idx, :]
            hist = hist.masked_fill(hist_pad.unsqueeze(-1), 0.0)
            BA, T = B * A, hist.size(2)
            hist2d = hist.view(BA, T, d)
            pad2d = hist_pad.view(BA, T)
            pe = self._build_sinusoidal_pe(T, d, hist2d.device)
            hist2d = hist2d + pe.unsqueeze(0)
            hist_out, _ = self.hist_self_attn(hist2d, hist2d, hist2d, key_padding_mask=pad2d, need_weights=False)
            lengths = (~pad2d).sum(dim=1)
            last_idx = torch.clamp(lengths - 1, min=0)
            arange_ba = torch.arange(BA, device=enc_nodes.device)
            hist_summary = hist_out[arange_ba, last_idx, :].view(B, A, d)
            hist_summary = self.hist_norm(hist_summary)
        else:
            hist_summary = torch.zeros(B, A, d, device=enc_nodes.device, dtype=enc_nodes.dtype)

        # 2) 跨-agent 历史交互：让每个 agent 感知他人近期意图
        #    将 enc_agents 与上一步得到的 hist_summary 融合为 agent_ctx，
        #    在 agent 维度上做一次自注意力与 FFN 残差堆叠，得到跨-agent 强化后的“历史摘要”。
        agent_ctx = torch.cat([enc_agents, hist_summary], dim=-1)   # [B,A,2d]
        agent_ctx = self.agent_ctx_proj(agent_ctx)                   # [B,A,d]
        agent_ctx = self.agent_ctx_norm(agent_ctx)
        a_out, _ = self.agent_interact_attn(agent_ctx, agent_ctx, agent_ctx, need_weights=False)
        agent_ctx = self.agent_interact_norm(agent_ctx + a_out)
        a_ffn = self.agent_interact_ffn(agent_ctx)
        agent_ctx = self.agent_interact_ffn_norm(agent_ctx + a_ffn)  # [B,A,d]
        # 用跨-agent 交互后的摘要替换原先的 hist_summary
        hist_summary = agent_ctx

        # 3) 与 depot 拼接后过 MLP -> context（每个 agent 一个 context）
        depot_expand = enc_depot.expand(-1, A, -1)               # [B, A, d]
        context = torch.cat([enc_agents, depot_expand, hist_summary], dim=-1)    # [B, A, 3d]
        context = self.ctx_proj(context)                         # [B, A, d]
        context = self.ctx_norm(context)

        if torch.isnan(enc_nodes).any() or torch.isinf(enc_nodes).any():
            print(f"[ERROR] nodes contains NaN/Inf")

        # 4) 标准 Cross-Attn：Q=context, K/V=kv
        kv = torch.cat([enc_depot, enc_nodes], dim=1)            # [B,1+N,d]
        kv_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=node_mask.device),  # depot 不屏蔽
            node_mask
        ], dim=1)                                                # [B,1+N]
        attn_out, _ = self.cross_attn(context, kv, kv, key_padding_mask=kv_mask, need_weights=False)
        context = self.cross_attn_norm(context + attn_out)       # Residual 1
        ffn_out = self.cross_ffn(context)
        context = self.cross_ffn_norm(context + ffn_out)         # Residual 2
        if torch.isnan(context).any() or torch.isinf(context).any():
            print(f"[ERROR][cross-block] context contains NaN/Inf")

        # 5) 打分（未归一化 logits）：scores = (Q K^T) / sqrt(d)
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