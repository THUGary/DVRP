"""Adversarial training package.

This package provides utilities to build environments, planners and a
diffusion-based generator for adversarial training experiments, plus a
trainer class that performs RL-style adversarial updates on the generator.

Public symbols are re-exported here for convenient imports, e.g.:

	from adversarial import build_env, DiffusionAdversarialTrainer

"""

from .types import EpisodeResult, GeneratorPolicy, PlannerPolicy, DemandTuple
from .builders import build_env, build_planner, build_diffusion
from .co_train.train_coevolution import coevolution_loop, CoevolutionConfig

__all__ = [
	"EpisodeResult",
	"GeneratorPolicy",
	"PlannerPolicy",
	"DemandTuple",
	"build_env",
	"build_planner",
	"build_diffusion",
	"coevolution_loop",
	"CoevolutionConfig",
]
