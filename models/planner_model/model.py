from __future__ import annotations
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder


def prepare_features(
    *,
    nodes,
    node_mask,
    depot,
    d_model: int = 128,
    device: str | torch.device = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    将 Encoder 需要的量（nodes/node_mask/depot）转换为张量。

    返回:
      {
        "nodes":     [B, N, 5] float32,
        "node_mask": [B, N]    bool,
        "depot":     [B, 1, 3] float32
      }
    """
    dev = torch.device(device)

    # nodes
    if isinstance(nodes, torch.Tensor):
        nodes_t = nodes.to(dev)
        if nodes_t.dim() == 2:
            nodes_t = nodes_t.unsqueeze(0)
    else:
        nodes_t = torch.tensor(nodes, dtype=torch.float32, device=dev)
        if nodes_t.dim() == 2:
            nodes_t = nodes_t.unsqueeze(0)  # [1, N, 5]

    # node_mask
    if isinstance(node_mask, torch.Tensor):
        mask_t = node_mask.to(dev)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)
    else:
        mask_t = torch.tensor(node_mask, dtype=torch.bool, device=dev)
        if mask_t.dim() == 1:
            mask_t = mask_t.unsqueeze(0)    # [1, N]

    # depot
    if isinstance(depot, torch.Tensor):
        depot_t = depot.to(dev)
        if depot_t.dim() == 2:
            depot_t = depot_t.unsqueeze(0)
    else:
        depot_t = torch.tensor(depot, dtype=torch.float32, device=dev)
        if depot_t.dim() == 2:
            depot_t = depot_t.unsqueeze(0)  # [1, 1, 3]

    out = {"nodes": nodes_t, "node_mask": mask_t, "depot": depot_t}

    return out


def prepare_agents(agents, device: str | torch.device = "cpu") -> torch.Tensor:
    """
    将 agents 转为张量（只做形状与 dtype 规整，不做特征工程）。
    返回 [B, A, 4] float32，其中四维约定为 (x, y, s, t_agent) 。
    """
    dev = torch.device(device)
    if isinstance(agents, torch.Tensor):
        t = agents.to(dev)
        if t.dim() == 2:
            t = t.unsqueeze(0)  # [1, A, 4]
        return t
    else:
        t = torch.tensor(agents, dtype=torch.float32, device=dev)
        if t.dim() == 2:
            t = t.unsqueeze(0)
        return t


class DVRPNet(nn.Module):
    """
      - encode(feats)        仅编码 nodes/depot
      - decode_step(feats, lateness_lambda, current_time) 
    并在内部（若提供）使用 feats['agents'] 参与解码；
    输出 [B, N+1] logits（第 0 列为 depot，1..N 为 nodes）。
    """

    def __init__(self, d_model: int = 128, nhead: int = 8, nlayers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(d_model=d_model, nhead=nhead, nlayers=nlayers)
        self.decoder = Decoder(d_model=d_model, nhead=nhead)

    @torch.no_grad()
    def encode(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        仅编码 nodes/depot；返回:
          {
            "H_nodes": [B, N, d],
            "H_depot": [B, 1, d],
            "node_mask": [B, N] (bool)
          }
        """
        H_nodes, H_depot, node_mask = self.encoder(feats)
        return {"H_nodes": H_nodes, "H_depot": H_depot, "node_mask": node_mask}

    @torch.no_grad()
    def encode_agents(self, agents_tensor: torch.Tensor) -> torch.Tensor:
        """将 agents 状态编码为 [B, A, d]。
        """
        return self.encoder.encode_agents(agents_tensor)

    def decode(
        self,
        *,
        enc_nodes: torch.Tensor,
        enc_depot: torch.Tensor,
        node_mask: torch.Tensor,
        enc_agents: torch.Tensor,
        agents_tensor: Optional[torch.Tensor] = None,
        nodes: Optional[torch.Tensor] = None,
        lateness_lambda: float = 0.0,
        history_indices: Optional[torch.Tensor] = None,
        history_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        仅执行解码一步：输入为 encode 后的张量与当前 agents 状态。
        参数：
          - enc_nodes: [B, N, d]
          - enc_depot: [B, 1, d]
          - node_mask: [B, N] (bool) True=屏蔽/已选
          - agents_tensor: [B, A, 4]  (x, y, s, t_agent)
          - nodes: [B, N, 5]（可选，仅用于迟到惩罚计算）
                返回：
                    - logits: [B, A, N+1]，按 [depot, nodes...]
        """
        logits = self.decoder(
            enc_nodes=enc_nodes,
            enc_depot=enc_depot,
            node_mask=node_mask,
            enc_agents=enc_agents,
            history_indices=history_indices,
            history_positions=history_positions,
        )  # [B, A, N+1]

        # 若提供 nodes，可对 nodes 段施加时间窗惩罚与容量可行性屏蔽
        if nodes is not None and agents_tensor is not None:
            B, N, _ = nodes.shape
            A = logits.size(1)
            # 节点段是 logits[..., 1:1+N]
            logits_nodes = logits[..., 1:1+N]

            # 1) 迟到惩罚（soft TW）
            if lateness_lambda and lateness_lambda > 0:
                ax = agents_tensor[..., 0].unsqueeze(-1)  # [B,A,1]
                ay = agents_tensor[..., 1].unsqueeze(-1)  # [B,A,1]
                nx = nodes[..., 0].unsqueeze(1)          # [B,1,N]
                ny = nodes[..., 1].unsqueeze(1)          # [B,1,N]
                t_due = nodes[..., 4].unsqueeze(1)       # [B,1,N]
                dist = (ax - nx).abs() + (ay - ny).abs()  # [B,A,N]
                t_agent = agents_tensor[..., 3].unsqueeze(-1)  # [B,A,1]
                eta = t_agent + dist                             # [B,A,N]
                lateness = torch.clamp(eta - t_due, min=0.0)     # [B,A,N]
                logits_nodes = logits_nodes - lateness_lambda * lateness

            # 2) 容量约束：需求大于剩余容量的节点置为 -inf
            demand = nodes[..., 3].unsqueeze(1)                  # [B,1,N]
            cap = agents_tensor[..., 2].unsqueeze(-1)            # [B,A,1]
            infeasible = demand > cap                            # [B,A,N]
            neg_inf = torch.finfo(logits.dtype).min
            logits_nodes = logits_nodes.masked_fill(infeasible, neg_inf)

            logits = torch.cat([logits[..., 0:1], logits_nodes], dim=-1)

        return logits

    @staticmethod
    def _manhattan(a_xy: torch.Tensor, b_xy: torch.Tensor) -> torch.Tensor:
        """a_xy: [...,2], b_xy: [...,2] -> Manhattan distance [...]
        """
        return (a_xy[..., 0] - b_xy[..., 0]).abs() + (a_xy[..., 1] - b_xy[..., 1]).abs()

    def forward(
        self,
        *,
        feats: Dict[str, torch.Tensor],
        agents: torch.Tensor,        # [B, A, 4]
        k: int,                      # 规划未来 k 步
        lateness_lambda: float = 0.0,
        cap_full: torch.Tensor,       # [B,A]，必须由构造时提供的 full_capacity 指定（来自 Config.capacity）
    ) -> Dict[str, torch.Tensor]:
        """
        流程：
          1) 对静态信息（nodes/depot/mask）编码一次
          2) 重复 k 次：对所有 agent 同步解码一步，选 argmax；将选中的节点加入全局 mask；
             并基于选中目标更新每个 agent 的 (x,y,t)。
        输入：
          feats: {nodes[B,N,5], node_mask[B,N](bool), depot[B,1,3]}
          agents: [B,A,4] (x,y,s,t)
                输出：
                    { "indices": [B,A,k] (每步选择的索引，0 表示 depot，1..N 表示 nodes[0..N-1]),
            "coords":  [B,A,k,2] (每步目标坐标) }
        """
        assert k >= 1, "k must be >= 1"
        nodes = feats["nodes"]              # [B,N,5]
        node_mask0 = feats["node_mask"]     # [B,N]
        depot = feats["depot"]              # [B,1,3]

        # 编码静态
        enc = self.encode(feats)
        Hn, Hd, mask = enc["H_nodes"], enc["H_depot"], enc["node_mask"].clone()

        B, N, _ = nodes.shape
        A = agents.size(1)

        # 状态拷贝（避免原地覆盖调用者张量）
        ag = agents.clone()  # [B,A,4]
        # cap_full must be provided by caller (来自 Config.capacity)。禁止退化为 agent 初始 s
        if cap_full is None:
            raise RuntimeError("cap_full (full capacity) must be provided to DVRPNet.forward — no fallback to agent initial s. Pass Config.capacity as cap_full.")
        cap_full_local = cap_full.clone()
        out_idx = torch.full((B, A, k), 0, dtype=torch.long, device=nodes.device)
        out_xy = torch.zeros(B, A, k, 2, dtype=torch.long, device=nodes.device)

        # 预取坐标
        node_xy = nodes[..., :2].long()      # [B,N,2]
        depot_xy = depot[..., :2].long().squeeze(1)  # [B,2]

        for step in range(k):
            # 单次 decode，随后基于置信度贪心分配，避免同一步冲突
            enc_agents = self.encoder.encode_agents(ag)
            logits = self.decode(
                enc_nodes=Hn,
                enc_depot=Hd,
                node_mask=mask,
                enc_agents=enc_agents,
                agents_tensor=ag,
                nodes=nodes,
                lateness_lambda=lateness_lambda,
                history_indices=None,
            )  # [B,A,N+1]

            sel = torch.full((B, A), N, dtype=torch.long, device=nodes.device)
            dest_xy = torch.zeros(B, A, 2, dtype=torch.long, device=nodes.device)

            for b in range(B):
                lb = logits[b]  # [A,N+1]
                order = torch.argsort(lb, dim=-1, descending=True)  # [A,N+1]
                # used_nodes 标记 nodes 是否被占用（不包括 depot）
                used = torch.zeros(N, dtype=torch.bool, device=nodes.device)
                assigned = torch.zeros(A, dtype=torch.bool, device=nodes.device)
                ptr = torch.zeros(A, dtype=torch.long, device=nodes.device)
                assigned_cnt = 0

                while assigned_cnt < A:
                    # 为每个未分配 agent 选出当前最优可用候选
                    candidates = torch.full((A,), N, dtype=torch.long, device=nodes.device)
                    c_scores = torch.full((A,), -float("inf"), device=nodes.device)
                    for aidx in range(A):
                        if assigned[aidx]:
                            continue
                        pa = int(ptr[aidx].item())
                        while pa < N + 1:
                            idx = int(order[aidx, pa].item())
                            # depot=0 可重复；nodes=1..N 需唯一
                            if idx == 0 or (1 <= idx <= N and not bool(used[idx-1].item())):
                                candidates[aidx] = idx
                                c_scores[aidx] = lb[aidx, idx]
                                ptr[aidx] = torch.tensor(pa, device=nodes.device)
                                break
                            pa += 1
                    # 选择全局最优 (agent, idx)
                    best_a = int(torch.argmax(c_scores).item())
                    best_idx = int(candidates[best_a].item())
                    sel[b, best_a] = best_idx
                    assigned[best_a] = True
                    assigned_cnt += 1
                    if 1 <= best_idx <= N:
                        used[best_idx - 1] = True

                # 写入目的地坐标
                for aidx in range(A):
                    idx = int(sel[b, aidx].item())
                    if 1 <= idx <= N:
                        dest_xy[b, aidx] = node_xy[b, idx - 1]
                    else:
                        dest_xy[b, aidx] = depot_xy[b]

            # 记录输出
            out_idx[..., step] = sel
            out_xy[..., step, :] = dest_xy

            # 更新全局 mask/agents（基于选择结果）
            if N > 0:
                for b in range(B):
                    for aidx in range(A):
                        idx = int(sel[b, aidx].item())
                        if 1 <= idx <= N:
                            mask[b, idx - 1] = True

            # 更新 agent 的时间、位置与容量（曼哈顿距离 + 扣减需求；depot 则恢复满容量）
            cur_xy = ag[..., :2].long()            # [B,A,2]
            dist = self._manhattan(cur_xy, dest_xy).to(ag.dtype)  # [B,A]
            ag[..., 0:2] = dest_xy.to(ag.dtype)
            ag[..., 3] = ag[..., 3] + dist
            if N > 0:
                for b in range(B):
                    for aidx in range(A):
                        idx = int(sel[b, aidx].item())
                        if 1 <= idx <= N:
                            d = nodes[b, idx - 1, 3].to(ag.dtype)
                            ag[b, aidx, 2] = torch.clamp(ag[b, aidx, 2] - d, min=0.0)
                        else:
                            ag[b, aidx, 2] = cap_full_local[b, aidx]

        return {"indices": out_idx, "coords": out_xy}

    # 兼容旧调用：decode_step(feats, ...) -> [B,N+1]
    def decode_step(
        self,
        feats: Dict[str, torch.Tensor],
        lateness_lambda: float = 0.0,
        current_time: float | int = 0,
    ) -> torch.Tensor:
        if "agents" not in feats:
            raise ValueError("decode_step requires 'agents' tensor in feats; include feats['agents'] of shape [B,A,4].")
        enc = self.encode(feats)
        agents_tensor = feats["agents"]
        enc_agents = self.encoder.encode_agents(agents_tensor)
        logits = self.decode(
            enc_nodes=enc["H_nodes"],
            enc_depot=enc["H_depot"],
            node_mask=enc["node_mask"],
            enc_agents=enc_agents,
            agents_tensor=agents_tensor,
            nodes=feats.get("nodes"),
            lateness_lambda=lateness_lambda,
            history_indices=None,
        )  # [B,A,N+1]
        # 历史用法假定 A==1 并返回 [B,N+1]
        if logits.size(1) == 1:
            return logits.squeeze(1)
        else:
            # 若 A>1，仍返回第一个 agent 的 logits 以维持旧行为
            return logits[:, 0, :]