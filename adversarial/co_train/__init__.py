"""Co-training (coevolution) utilities for planner and generator.

This subpackage provides a lightweight framework to iteratively train the
planner model and the diffusion demand generator in alternating cycles.

Key ideas:
  1. Maintain a registry of historical generator checkpoints (versions).
  2. When (re)training the planner, sample demands from all historical
     generator versions for robustness / fictitious self-play.
  3. When adversarially updating the generator, train against the latest
     planner (and optionally evaluate against older planners to avoid
     catastrophic forgetting).

The provided code offers structural scaffolding & hooks; domain-specific
losses and data collection logic can be integrated where indicated.
"""

from .version_registry import GeneratorVersion, GeneratorVersionRegistry
from .match_scheduler import PlannerTrainingScheduler
from .train_coevolution import coevolution_loop, CoevolutionConfig

__all__ = [
    "GeneratorVersion",
    "GeneratorVersionRegistry",
    "PlannerTrainingScheduler",
   "coevolution_loop",
   "CoevolutionConfig",
]
