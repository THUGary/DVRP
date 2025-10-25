from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset


def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class DVRPSyntheticDataset(Dataset):
    """
    合成 DVRP 单步监督数据：
    - 每条样本仅包含单个 agent 的当前时刻状态与一组候选节点（含 depot）
    - 标签是单步“下一访问目标”的分类（0..N 对应 N 为 depot）
    - 采用启发式：score = distance + lambda_late * lateness（lateness = max(0, ETA - t_due)）
      从可行（容量、出现时间）节点中选择最小 score 为标签；若无可行节点 -> 选择 depot

    特征结构：
      nodes: Tensor [N, 5] -> (x, y, t_arrival, demand, t_due)
      node_mask: Bool [N]  -> True 表示不可选（容量不够、尚未出现等）
      agent: Tensor [1, 4] -> (x, y, capacity, t_agent)
      depot: Tensor [1, 3] -> (x, y, t_agent)
      label: int in [0..N]  (N 代表 depot)
    """

    def __init__(
        self,
        size: int = 20000,
        grid_w: int = 20,
        grid_h: int = 20,
        max_nodes: int = 30,
        min_nodes: int = 5,
        max_demand: int = 3,
        max_capacity: int = 10,
        t_horizon: int = 50,
        due_min_slack: int = 3,
        due_max_slack: int = 15,
        lambda_late: float = 1.0,
        seed: Optional[int] = 42,
        depot_xy: Tuple[int, int] = (0, 0),
    ) -> None:
        super().__init__()
        self.size = size
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.max_demand = max_demand
        self.max_capacity = max_capacity
        self.t_horizon = t_horizon
        self.due_min_slack = due_min_slack
        self.due_max_slack = due_max_slack
        self.lambda_late = lambda_late
        self.depot_xy = depot_xy
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.size

    def _rand_int(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def _sample_instance(self) -> Dict:
        # 随机节点数量
        N = self._rand_int(self.min_nodes, self.max_nodes)
        # 当前全局时间
        t_now = self._rand_int(0, self.t_horizon // 2)

        # 生成节点
        nodes: List[Tuple[int, int, int, int, int]] = []
        for _ in range(N):
            x = self._rand_int(0, self.grid_w - 1)
            y = self._rand_int(0, self.grid_h - 1)
            t_arrival = self._rand_int(0, t_now)  # 已出现
            demand = self._rand_int(1, self.max_demand)
            slack = self._rand_int(self.due_min_slack, self.due_max_slack)
            t_due = t_arrival + slack
            nodes.append((x, y, t_arrival, demand, t_due))

        # agent
        ax = self._rand_int(0, self.grid_w - 1)
        ay = self._rand_int(0, self.grid_h - 1)
        cap = self._rand_int(1, self.max_capacity)
        agent_t = t_now

        depot_x, depot_y = self.depot_xy

        # mask：容量不够、尚未出现（t_arrival > t_now）的节点屏蔽
        node_mask: List[bool] = []
        for (x, y, t_arrival, demand, t_due) in nodes:
            mask = (demand > cap) or (t_arrival > t_now)
            node_mask.append(mask)

        # 计算标签
        best_score = float("inf")
        best_idx = -1
        for i, (x, y, t_arrival, demand, t_due) in enumerate(nodes):
            if node_mask[i]:
                continue
            dist = manhattan((ax, ay), (x, y))
            eta = agent_t + dist
            lateness = max(0, eta - t_due)
            score = dist + self.lambda_late * float(lateness)
            if score < best_score:
                best_score = score
                best_idx = i

        # 若没有任何可选节点，则选择 depot
        label = best_idx if best_idx >= 0 else len(nodes)

        return {
            "nodes": nodes,
            "node_mask": node_mask,
            "agent": (ax, ay, cap, agent_t),
            "depot": (depot_x, depot_y, agent_t),
            "label": label,
        }

    def __getitem__(self, idx: int) -> Dict:
        return self._sample_instance()


def dvrp_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    将变长节点的 batch pad 成统一长度（按本批最大 N）。
    输出：
      nodes:     [B, Nmax, 5]
      node_mask: [B, Nmax]  (True=masked)
      agents:    [B, 1, 4]
      depot:     [B, 1, 3]
      labels:    [B]  (范围 [0..N_i]，其中 N_i 变长；在 pad 后，超过实际 N_i 的 label 仍表示 depot 的位置：N_i)
    实现中我们把 depot 的索引固定为样本自身的 N_i，训练时会动态构建每个样本的有效长度并将无效位置 logits 置 -inf。
    """
    B = len(batch)
    maxN = max(len(item["nodes"]) for item in batch)
    device = torch.device("cpu")

    nodes = torch.zeros(B, maxN, 5, dtype=torch.float32)
    node_mask = torch.ones(B, maxN, dtype=torch.bool)  # 先全 True，再对有效长度置为各自 mask
    agents = torch.zeros(B, 1, 4, dtype=torch.float32)
    depot = torch.zeros(B, 1, 3, dtype=torch.float32)
    labels = torch.zeros(B, dtype=torch.long)
    valid_N = torch.zeros(B, dtype=torch.long)

    for b, item in enumerate(batch):
        Ni = len(item["nodes"])
        valid_N[b] = Ni
        if Ni > 0:
            nodes[b, :Ni] = torch.tensor(item["nodes"], dtype=torch.float32)
            mask_i = torch.tensor(item["node_mask"], dtype=torch.bool)
            node_mask[b, :Ni] = mask_i
        agents[b, 0] = torch.tensor(item["agent"], dtype=torch.float32)
        depot[b, 0] = torch.tensor(item["depot"], dtype=torch.float32)
        labels[b] = item["label"]

    return {
        "nodes": nodes,
        "node_mask": node_mask,
        "agents": agents,
        "depot": depot,
        "labels": labels,
        "valid_N": valid_N,
    }