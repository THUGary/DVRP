from __future__ import annotations
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MLP
from .encoder import NodeEncoder, DepotEncoder
from .decoder import ContextDecoder
from agent.controller.distance import travel_time  # 用于 ETA 软惩罚时的近似（解码时也会在planner里用）


class DVRPNet(nn.Module):
    """
    最小可行的 DVRP 模型实现：
    - NodeEncoder: nodes->nodes Self-Attn
    - DepotEncoder: depot<-nodes Cross-Attn（不回写到nodes）
    - Decoder: 融合(agent, depot) -> 对 [nodes, depot] 产生日志its（单步）
    训练时可重复 decode_step；推理时由 planner 循环调用 decode_step 完成多轮解码。
    """
    def __init__(self, d_model: int = 128, nhead: int = 8, nlayers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.node_mlp = MLP(5, d_model)      # (x, y, t_arrival, demand, t_due)
        self.agent_mlp = MLP(4, d_model)     # (x, y, capacity, t_agent)
        self.depot_mlp = MLP(3, d_model)     # (x, y, t_agent)

        self.node_encoder = NodeEncoder(d_model, nhead, nlayers)
        self.depot_encoder = DepotEncoder(d_model, nhead, 1)
        self.decoder = ContextDecoder(d_model, nhead, 1)

    def encode(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        feats:
          nodes: [B,N,5]
          agents: [B,1,4]  (planner单步对单agent decode)
          depot: [B,1,3]
          node_mask: [B,N] (True=masked)
        """
        B, N, _ = feats["nodes"].shape
        nodes_emb = self.node_mlp(feats["nodes"])
        agents_emb = self.agent_mlp(feats["agents"])
        depot_emb = self.depot_mlp(feats["depot"])

        H_nodes = self.node_encoder(nodes_emb)        # [B,N,D]
        H_depot = self.depot_encoder(depot_emb, H_nodes)  # [B,1,D]
        return {"H_nodes": H_nodes, "H_depot": H_depot, "agent_embed": agents_emb, "depot_embed": depot_emb}

    def decode_step(self, feats: Dict[str, torch.Tensor], lateness_lambda: float = 0.0, current_time: int = 0) -> torch.Tensor:
        """
        单步解码，返回 logits over [nodes, depot]。
        可选：对节点加上 -lateness_lambda * lateness 的偏置（软惩罚），不屏蔽 depot。
        feats 同 encode 输入
        """
        enc = self.encode(feats)
        logits = self.decoder(enc["agent_embed"], enc["depot_embed"], enc["H_nodes"], enc["H_depot"])  # [B,N+1]

        if lateness_lambda > 0.0:
            # 计算每个节点的预计迟到（粗略：使用 agent->node 的 travel_time 近似 ETA）
            # feats["nodes"]中 (x,y,t_arrival,demand,t_due)
            B, N, _ = feats["nodes"].shape
            agent_xy = feats["agents"][..., :2]  # [B,1,2]
            node_xy = feats["nodes"][..., :2]    # [B,N,2]
            # 简单 L1 距离近似
            # dt = |ax-x| + |ay-y|
            dt = torch.abs(agent_xy[..., 0].unsqueeze(-1) - node_xy[..., 0]) + torch.abs(agent_xy[..., 1].unsqueeze(-1) - node_xy[..., 1])  # [B,N]
            t_due = feats["nodes"][..., 4]  # [B,N]
            eta = current_time + dt
            lateness = torch.clamp(eta - t_due, min=0.0)  # [B,N]
            # 扩展到 [B,N+1]，depot 不惩罚
            lat_full = torch.cat([lateness, torch.zeros_like(lateness[:, :1])], dim=1)
            logits = logits - lateness_lambda * lat_full

        return logits


def prepare_features(
    nodes: List[Tuple[int, int, int, int, int]],
    node_mask: List[bool],
    agents: List[Tuple[int, int, int, int]],  # 单 agent: (x, y, capacity, t_agent)
    depot: Tuple[int, int, int],
    d_model: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    将 Python 列表特征转为张量，并满足 DVRPNet 的输入要求。
    """
    import torch
    # B=1（每次只对一个 agent 解码）
    if len(nodes) == 0:
        # 兜底：如果没有节点，构造一个 shape 正确的全零 placeholder
        nodes_t = torch.zeros(1, 0, 5, device=device)
        node_mask_t = torch.zeros(1, 0, dtype=torch.bool, device=device)
    else:
        nodes_t = torch.tensor(nodes, dtype=torch.float32, device=device).unsqueeze(0)  # [1,N,5]
        node_mask_t = torch.tensor(node_mask, dtype=torch.bool, device=device).unsqueeze(0)  # [1,N]

    ax, ay, cap, ta = agents[0]
    agents_t = torch.tensor([[ax, ay, cap, ta]], dtype=torch.float32, device=device).unsqueeze(1)  # [1,1,4]

    dx, dy, dt = depot
    depot_t = torch.tensor([[dx, dy, dt]], dtype=torch.float32, device=device).unsqueeze(1)       # [1,1,3]

    return {"nodes": nodes_t, "node_mask": node_mask_t, "agents": agents_t, "depot": depot_t}