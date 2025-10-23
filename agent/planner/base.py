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
    - plan(observations, agent_states, depot, t, horizon) -> list[deque[Target]]
    """

    def __init__(self, **params) -> None:
        self.params = params

    @abstractmethod
    def plan(
        self,
        observations: List[Tuple[int, int, int, int]],  # [(x,y,t,c), ...]
        agent_states: List[AgentState],
        depot: Tuple[int, int],
        t: int,
        horizon: int = 1,
    ) -> List[Deque[Target]]:
        raise NotImplementedError
