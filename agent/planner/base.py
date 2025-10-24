from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional
from collections import deque

@dataclass(frozen=True)
class AgentState:
    x: int
    y: int
    s: int  # remaining capacity/space

# Target is a coordinate (x, y). In a fuller impl, could be demand IDs.
Target = Tuple[int, int]

class BasePlanner(ABC):
    """Planner interface.

    Contract:
    - plan(observations, agent_states, depot, t, horizon, current_plans, global_nodes, serve_mark, unserved_count) 
      -> list[deque[Target]]
    
    节点数据结构: (x, y, t_arrival, t_due, demand)
    """

    def __init__(self, **params) -> None:
        self.params = params

    @abstractmethod
    def plan(
        self,
        observations: List[Tuple[int, int, int, int, int]],  # [(x, y, t_arrival, c, t_due), ...]
        agent_states: List[AgentState],
        depot: Tuple[int, int],
        t: int,
        horizon: int = 1,
        current_plans: Optional[List[Deque[Target]]] = None,  # 当前规划结果（上次规划的路径）
        global_nodes: Optional[List[Tuple[int, int, int, int, int]]] = None,  # 全局节点列表 [(x, y, t_arrival, t_due, demand), ...]
        serve_mark: Optional[List[int]] = None,  # 服务标记向量
        unserved_count: Optional[int] = None,  # 未服务节点数量
    ) -> List[Deque[Target]]:
        raise NotImplementedError