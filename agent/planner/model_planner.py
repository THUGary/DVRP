from __future__ import annotations
from typing import Deque, List, Tuple, Optional, Dict
from collections import deque

from .base import BasePlanner, AgentState, Target
from agent.controller.distance import travel_time

import torch
from models.planner_model import DVRPNet, prepare_features
_TORCH_OK = True

# 尝试导入 torch，若不可用则使用启发式回退
# try:
#     import torch
#     from models import DVRPNet, prepare_features
#     _TORCH_OK = True
# except Exception:
#     _TORCH_OK = False


class ModelPlanner(BasePlanner):
    """
    使用学习模型进行动态规划的 Planner。
    - 当需要重新规划时，编码一次，然后顺序解码多轮，直到所有 agent 的时间 t_i > base_t + time_plan
    - 若无需重新规划，则可直接返回 current_plans（由调用方判断）
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
        if current_plans is None:
            current_plans = [deque() for _ in range(num_agents)]

        # 构建候选节点列表（按当前可见）
        nodes: List[Tuple[int, int, int, int, int]] = list(observations)

        # 若 torch 不可用，用启发式回退（贪心最近邻，顺序解码）
        if not _TORCH_OK or self._model is None:
            return self._heuristic_plan(nodes, agent_states, depot, t)

        # 模型驱动规划
        # 初始化mask：节点唯一访问。mask[i]=True 表示第 i 个节点不可选
        N = len(nodes)
        mask = [False] * N  # depot 不在这个 mask 中

        # 初始化每个agent的时间与位置（各agent时间起点为全局 t）
        agent_times = [t for _ in range(num_agents)]
        agent_pos = [(a.x, a.y) for a in agent_states]
        agent_cap = [a.s for a in agent_states]

        # 规划结果（队列）
        out_plans: List[Deque[Target]] = [deque() for _ in range(num_agents)]

        # 截止时间：本次规划覆盖的窗口
        plan_until = t + max(1, int(self.time_plan))

        # 连续多轮，每轮为每个agent选一个下一目标
        # 直到所有 agent 的时间超过 plan_until，或没有可选节点
        while True:
            # 终止条件1：所有 agent 的时间超过 plan_until
            if all(at > plan_until for at in agent_times):
                break
            # 终止条件2：所有节点都被选完
            if all(mask_i for mask_i in mask) or N == 0:
                break

            # 逐个 agent 顺序解码一次
            for i in range(num_agents):
                # 若该 agent 的时间已超过 plan_until，则跳过
                if agent_times[i] > plan_until:
                    continue

                # 在剩余容量为0时，允许选择 depot，以便回仓；也可直接跳过
                # 准备特征张量
                with torch.no_grad():
                    feats = prepare_features(
                        nodes=nodes,
                        node_mask=mask,
                        agents=[(agent_pos[i][0], agent_pos[i][1], agent_cap[i], agent_times[i])],
                        depot=(depot[0], depot[1], agent_times[i]),  # depot 特征里放入当前 agent 时间
                        d_model=self.d_model,
                        device=self.device,
                    )
                    # 模型前向，得到 logits over [nodes, depot]，其中 depot 固定在最后一个位置（或第0维）
                    # 此处我们将 candidates = [nodes..., depot]，并在内部保证 depot 不mask
                    logits = self._model.decode_step(
                        feats,
                        lateness_lambda=self.lateness_lambda,
                        current_time=agent_times[i],
                    )  # shape [1, N+1]

                    # 构造mask: 节点按 mask[i] True 则屏蔽；depot 不屏蔽
                    node_mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)  # [1, N]
                    # 拼出 [nodes, depot] 的mask
                    full_mask = torch.cat([node_mask_tensor, torch.zeros(1, 1, dtype=torch.bool, device=self.device)], dim=1)  # [1, N+1]
                    logits = logits.masked_fill(full_mask, float("-inf"))
                    # 选择
                    sel_idx = int(torch.argmax(logits, dim=-1).item())  # 0..N 表示 N 为 depot

                # 解析选择
                if sel_idx == N:
                    # 选择了 depot
                    target = depot
                else:
                    target = (nodes[sel_idx][0], nodes[sel_idx][1])

                # 更新计划
                out_plans[i].append(target)

                # 更新时间、位置、容量与mask
                dt = travel_time(agent_pos[i], target)
                agent_times[i] += dt
                agent_pos[i] = target
                if sel_idx != N:
                    # 消耗容量
                    demand = nodes[sel_idx][3]  # (x,y,t_arrival,demand,t_due)
                    agent_cap[i] = max(0, agent_cap[i] - demand)
                    # 屏蔽该节点（唯一访问）
                    mask[sel_idx] = True

            # 若本轮所有 agent 都未能添加目标（极端情况），避免死循环
            if all(len(out_plans[i]) == 0 for i in range(num_agents)):
                break

        # 兜底：若某些 agent 仍没有目标，给一个 depot
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
        """
        torch 不可用时的简易启发式：顺序为每个 agent 选择距离最近的未访问节点，直到超过 time_plan。
        """
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
                # 找最近且可承载的
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
                    # 回仓
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