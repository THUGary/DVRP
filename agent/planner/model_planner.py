from __future__ import annotations
from typing import Deque, List, Tuple, Optional, Dict
from collections import deque

from .base import BasePlanner, AgentState, Target
from agent.controller.distance import travel_time

import torch
from models.planner_model import DVRPNet, prepare_features
_TORCH_OK = True


class ModelPlanner(BasePlanner):
    """
    使用学习模型进行动态规划的 Planner。
    - 多轮解码：每轮为每个 agent 选择一个目标；更新 mask/时间/容量
    - 直到所有 agent 的时间 t_i > base_t + time_plan 或无可选节点
    - 若无需重新规划（外层判断），可沿用 current_plans
    新增：
    - load_from_ckpt(ckpt_path): 载入已训练模型权重
    """

    def __init__(self, d_model: int = 128, nhead: int = 8, nlayers: int = 2, time_plan: int = 6,
                 lateness_lambda: float = 0.0, device: str = "cpu", **params) -> None:
        """
        lateness_lambda: 若 >0，会对 logits 添加 -lambda * lateness 的偏置（ETA>due 的软惩罚）
        time_plan: 每次重规划覆盖未来的仿真时间窗口
        """
        super().__init__(**params)
        self.d_model = d_model
        self.nhead = nhead
        self.nlayers = nlayers
        self.time_plan = time_plan
        self.lateness_lambda = lateness_lambda
        self.device = device

        self._model: Optional["DVRPNet"] = None
        if _TORCH_OK:
            self._model = DVRPNet(d_model=d_model, nhead=nhead, nlayers=nlayers).to(device)
            self._model.eval()

    def load_from_ckpt(self, ckpt_path: str) -> None:
        """从 checkpoints 加载 DVRPNet 权重。"""
        if self._model is None:
            self._model = DVRPNet(d_model=self.d_model, nhead=self.nhead, nlayers=self.nlayers).to(self.device)
        blob = torch.load(ckpt_path, map_location=self.device)
        state = blob.get("model", blob)  # 兼容只存 state_dict 或 dict{model:...}
        self._model.load_state_dict(state, strict=False)
        self._model.eval()

    def plan(
        self,
        observations: List[Tuple[int, int, int, int, int]],  # [(x,y,t_arrival,demand,t_due), ...]
        agent_states: List[AgentState],  # x,y,s
        depot: Tuple[int, int],
        t: int,
        horizon: int = 1,
        current_plans: Optional[List[Deque[Target]]] = None,
        global_nodes: Optional[List[Tuple[int, int, int, int, int]]] = None,
        serve_mark: Optional[List[int]] = None,
        unserved_count: Optional[int] = None,
    ) -> List[Deque[Target]]:
        """
        返回每个 agent 的目标队列（deque[(x,y), ...]）
        """
        num_agents = len(agent_states)

        # 候选节点列表（当前可见）
        nodes: List[Tuple[int, int, int, int, int]] = list(observations)

        # 若 torch 不可用，用启发式回退（贪心最近邻，顺序解码）
        if not _TORCH_OK or self._model is None:
            return self._heuristic_plan(nodes, agent_states, depot, t)

        # 初始化 mask/时间/位置/容量
        N = len(nodes)
        mask = [False] * N  # 节点唯一访问；depot 不在 mask 内
        agent_times = [t for _ in range(num_agents)]
        agent_pos = [(a.x, a.y) for a in agent_states]
        agent_cap = [a.s for a in agent_states]
        out_plans: List[Deque[Target]] = [deque() for _ in range(num_agents)]
        plan_until = t + max(1, int(self.time_plan))

        # 多轮解码：每轮每个 agent 选择一个候选
        while True:
            if all(at > plan_until for at in agent_times):
                break
            if all(mask_i for mask_i in mask) or N == 0:
                break

            for i in range(num_agents):
                if agent_times[i] > plan_until:
                    continue

                with torch.no_grad():
                    feats = prepare_features(
                        nodes=nodes,
                        node_mask=mask,
                        agents=[(agent_pos[i][0], agent_pos[i][1], agent_cap[i], agent_times[i])],
                        depot=(depot[0], depot[1], agent_times[i]),
                        d_model=self.d_model,
                        device=self.device,
                    )
                    logits = self._model.decode_step(
                        feats,
                        lateness_lambda=self.lateness_lambda,
                        current_time=agent_times[i],
                    )  # [1, N+1]
                    node_mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)  # [1,N]
                    full_mask = torch.cat([node_mask_tensor, torch.zeros(1, 1, dtype=torch.bool, device=self.device)], dim=1)
                    logits = logits.masked_fill(full_mask, float("-inf"))
                    sel_idx = int(torch.argmax(logits, dim=-1).item())

                if sel_idx == N:
                    target = depot
                else:
                    target = (nodes[sel_idx][0], nodes[sel_idx][1])

                out_plans[i].append(target)
                dt = travel_time(agent_pos[i], target)
                agent_times[i] += dt
                agent_pos[i] = target
                if sel_idx != N:
                    demand = nodes[sel_idx][3]
                    agent_cap[i] = max(0, agent_cap[i] - demand)
                    mask[sel_idx] = True

            # 防死循环：若本轮所有 agent 都未添加目标
            if all(len(q) == 0 for q in out_plans):
                break

        for i in range(num_agents):
            if len(out_plans[i]) == 0:
                out_plans[i].append(depot)

        return out_plans

    def _heuristic_plan(
        self,
        nodes: List[Tuple[int, int, int, int, int]],
        agent_states: List[AgentState],
        depot: Tuple[int, int],
        t: int,
    ) -> List[Deque[Target]]:
        """启发式回退：最近邻+容量约束，直到超过 time_plan。"""
        N = len(nodes)
        num_agents = len(agent_states)
        visited = [False] * N
        agent_times = [t for _ in range(num_agents)]
        agent_pos = [(a.x, a.y) for a in agent_states]
        agent_cap = [a.s for a in agent_states]
        plan_until = t + max(1, int(self.time_plan))
        out = [deque() for _ in range(num_agents)]

        while True:
            if all(at > plan_until for at in agent_times):
                break
            if all(visited) or N == 0:
                break
            for i in range(num_agents):
                if agent_times[i] > plan_until:
                    continue
                best_j, best_d = -1, 1e9
                for j in range(N):
                    if visited[j]:
                        continue
                    demand = nodes[j][3]
                    if agent_cap[i] < demand:
                        continue
                    d = abs(agent_pos[i][0] - nodes[j][0]) + abs(agent_pos[i][1] - nodes[j][1])
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j == -1:
                    out[i].append(depot)
                    agent_times[i] += travel_time(agent_pos[i], depot)
                    agent_pos[i] = depot
                else:
                    out[i].append((nodes[best_j][0], nodes[best_j][1]))
                    agent_times[i] += best_d
                    agent_pos[i] = (nodes[best_j][0], nodes[best_j][1])
                    agent_cap[i] -= nodes[best_j][3]
                    visited[best_j] = True

        for i in range(num_agents):
            if len(out[i]) == 0:
                out[i].append(depot)
        return out