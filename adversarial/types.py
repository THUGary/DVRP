from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, List, Tuple, Deque, Callable, Dict, Any, Optional

DemandTuple = Tuple[int,int,int,int,int]  # (x,y,t,c,end_t)
Action = Tuple[int,int]

@dataclass
class EpisodeResult:
    env_reward: float
    gen_reward: float
    steps: int
    served: int
    total_demands: int
    info: Dict[str, Any]

class GeneratorPolicy(Protocol):
    def sample_demands(self, cfg: Any, device: str) -> List[DemandTuple]: ...

class PlannerPolicy(Protocol):
    def plan(self, obs_demands: List[DemandTuple], agent_states, depot: Tuple[int,int], t: int, horizon: int = 1): ...

StepCallback = Callable[[int, Dict[str, Any]], None]
EpisodeCallback = Callable[[int, EpisodeResult], None]
