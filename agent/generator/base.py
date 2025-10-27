from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Sequence, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class Demand:
    x: int
    y: int
    t: int  # appearance time
    c: int  # quantity/demand
    end_t: int  # cancel time (exclusive): demand is canceled when current time > end_t

class BaseDemandGenerator(ABC):
    """Interface for demand generators.

    Contract:
    - reset(seed, width, height, **params) -> None
    - sample(t) -> list of Demands that appear at time t
    """

    width: int
    height: int

    def __init__(self, width: int, height: int, **params) -> None:
        self.width = width
        self.height = height
        self.params = params
        self.total_demand=int(params.get("total_demand",1))

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state. Implementations may use seed for RNG."""
        pass

    @abstractmethod
    def sample(self, t: int) -> List[Demand]:
        """Return the list of new demands that appear at time t."""
        raise NotImplementedError