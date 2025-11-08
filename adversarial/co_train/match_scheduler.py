from __future__ import annotations
"""Planner training scheduler for sampling generator versions.

Provides simple policies (uniform, latest-skewed) to choose which generator
version to sample per episode when building planner training data.
"""
import random
from typing import List
from .version_registry import GeneratorVersion, GeneratorVersionRegistry


class PlannerTrainingScheduler:
    def __init__(self, registry: GeneratorVersionRegistry, policy: str = "uniform", latest_bias: float = 0.7):
        self.registry = registry
        self.policy = policy
        self.latest_bias = latest_bias

    def pick(self, rng: random.Random) -> GeneratorVersion:
        versions: List[GeneratorVersion] = self.registry.list()
        if not versions:
            raise RuntimeError("No generator versions in registry")
        if self.policy == "uniform" or len(versions) == 1:
            return rng.choice(versions)
        if self.policy == "latest_biased":
            if rng.random() < self.latest_bias:
                return versions[-1]
            return rng.choice(versions[:-1])
        # default fallback
        return rng.choice(versions)
