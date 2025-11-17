from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Sequence, Optional
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class Demand:
    x: int
    y: int
    t: int  # appearance time
    c: int  # quantity/demand
    end_t: int  # cancel time (exclusive): demand is canceled when current time > end_t
    service_time: int = 0  # additional service duration required upon arrival

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
        # RNG used for sampling service times; seed set during reset when available
        self._service_rng = random.Random(params.get("service_time_seed"))
        # Depot coordinate (optional). When provided, generators may avoid producing demands at depot.
        depot = params.get("depot")
        try:
            self.depot = tuple(depot) if depot is not None else None
        except Exception:
            print("[Generator] Warning: no depot input, demand generator may produce demands at depot.")
            self.depot = None
        # Remaining total demand budget (for certain generators)
        self.total_demand = int(params.get("total_demand", 1))
        self.max_time = int(params.get("max_time", 0))

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal state. Implementations may use seed for RNG."""
        if seed is not None:
            self._service_rng.seed(seed)

    @abstractmethod
    def sample(self, t: int) -> List[Demand]:
        """Return the list of new demands that appear at time t."""
        raise NotImplementedError

    # --- Service-time helpers -------------------------------------------------
    def sample_service_time(self, *, capacity: Optional[int] = None) -> int:
        """Draw a discrete service time for a demand.

        Parameters
        ----------
        capacity: Optional[int]
            Demand size (capacity requirement). When provided and
            `service_time_per_unit` is configured (>0), the sampled value will be
            augmented linearly by this amount.
        """
        min_service = int(self.params.get("min_service_time", 0))
        max_service = int(self.params.get("max_service_time", min_service))
        if max_service < min_service:
            max_service = min_service
        if min_service == max_service:
            base = max(0, min_service)
        else:
            base = self._service_rng.randint(min_service, max_service)
        per_unit = float(self.params.get("service_time_per_unit", 0.0))
        if capacity is not None and per_unit > 0.0:
            base = base + int(round(per_unit * max(0, capacity)))
        return max(0, int(base))