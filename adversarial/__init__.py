"""Adversarial training package.

This package provides utilities to build environments, planners and a
diffusion-based generator for adversarial training experiments, plus a
trainer class that performs RL-style adversarial updates on the generator.

Public symbols are re-exported here for convenient imports, e.g.:

	from adversarial import build_env, DiffusionAdversarialTrainer

"""

from .types import EpisodeResult, GeneratorPolicy, PlannerPolicy, DemandTuple
from .builders import build_env, build_planner, build_diffusion
from training.generator.adversarial_trainer import (
	DiffusionAdversarialTrainer,
	AdvConfig,
	rollout_episode,
)

__all__ = [
	"EpisodeResult",
	"GeneratorPolicy",
	"PlannerPolicy",
	"DemandTuple",
	"build_env",
	"build_planner",
	"build_diffusion",
	"DiffusionAdversarialTrainer",
	"AdvConfig",
	"rollout_episode",
]
