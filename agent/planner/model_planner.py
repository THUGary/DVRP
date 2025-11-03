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
                 lateness_lambda: float = 0.0, device: str = "cpu", full_capacity: int | None = None, **params) -> None:
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
        # 满容量：若提供则在返回 depot 时将容量恢复到该值；否则退化为各 agent 初始 s
        self.full_capacity = full_capacity

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

        # 组装批次（B=1）与初始 mask
        N = len(nodes)

        # 如果节点数为0，直接返回所有 agent 回 depot，horizon个depot
        if N == 0:
            return [deque([depot] * horizon) for _ in range(num_agents)]

        mask = [False] * N
        agents_tensor = [
            (a.x, a.y, a.s, t) for a in agent_states
        ]  # [A,4]
        # cap_full: [1,A]，必须由构造时提供的 full_capacity 指定（来自 Config.capacity）
        if self.full_capacity is None:
            raise RuntimeError("ModelPlanner requires full_capacity (Config.capacity) at construction; pass full_capacity=cfg.capacity.")
        cap_full = torch.full((1, len(agent_states)), float(self.full_capacity), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            feats = prepare_features(
                nodes=[nodes],                 # [1,N,5]
                node_mask=[mask],              # [1,N]
                depot=[(depot[0], depot[1], t)],  # [1,1,3]
                d_model=self.d_model,
                device=self.device,
            )
            agents_t = torch.tensor([agents_tensor], dtype=torch.float32, device=self.device)  # [1,A,4]
            k = max(1, int(horizon))
            out = self._model.forward(
                feats=feats,
                agents=agents_t,
                k=k,
                lateness_lambda=self.lateness_lambda,
                cap_full=cap_full,  # 回 depot 恢复容量
            )

        # 解析输出到每个 agent 的 deque
        coords = out["coords"].squeeze(0)  # [A,k,2]
        out_plans: List[Deque[Target]] = [deque() for _ in range(num_agents)]
        for aidx in range(num_agents):
            for step in range(coords.size(1)):
                x, y = int(coords[aidx, step, 0].item()), int(coords[aidx, step, 1].item())
                out_plans[aidx].append((x, y))

        # 若某个 agent 未得到目标，至少回仓
        for i in range(num_agents):
            if len(out_plans[i]) == 0:
                out_plans[i].append(depot)

        return out_plans
