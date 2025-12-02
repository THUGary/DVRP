from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from .base import BaseDemandGenerator, Demand


class StaticDemandGenerator(BaseDemandGenerator):
    """Wraps a generator so every demand releases at t=0 and lives forever (no expiry)."""

    # Very large number to ensure demands never expire in static VRP
    INFINITE_END_TIME = 999999

    def __init__(self, base_generator: BaseDemandGenerator, *, max_end_time: Optional[int] = None) -> None:
        super().__init__(base_generator.width, base_generator.height, **getattr(base_generator, "params", {}))
        self._base = base_generator
        self._snapshot: List[Demand] = []
        self._released = False
        # For static VRP, demands should never expire - use a very large end_t
        # Ignore max_end_time parameter as it's not relevant for static VRP
        self._extended_end_t = self.INFINITE_END_TIME

    def reset(self, seed: Optional[int] = None) -> None:
        # IMPORTANT: Set global random states before generating demands.
        # The base generator and its neighborhoods may use global random/np.random
        # internally, so we must ensure reproducibility by seeding them here.
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self._base.reset(seed)
        self._snapshot = []
        max_time = int(getattr(self._base, "max_time", self.params.get("max_time", 1)))
        max_time = max(1, max_time)
        for t in range(max_time):
            demands_t = self._base.sample(t)
            if not demands_t:
                continue
            for demand in demands_t:
                end_t = int(max(self._extended_end_t, int(getattr(demand, "end_t", 0))))
                static_demand = Demand(
                    x=int(demand.x),
                    y=int(demand.y),
                    t=0,
                    c=int(demand.c),
                    end_t=end_t,
                    service_time=int(getattr(demand, "service_time", 0)),
                )
                self._snapshot.append(static_demand)
        self._released = False

    def sample(self, t: int) -> List[Demand]:
        if t == 0 and not self._released:
            self._released = True
            return list(self._snapshot)
        return []


__all__ = ["StaticDemandGenerator"]
