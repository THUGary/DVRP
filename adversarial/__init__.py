"""Adversarial training package.

This package provides utilities to build environments, planners and a
diffusion-based generator for adversarial training experiments, plus a
trainer class that performs RL-style adversarial updates on the generator.

Updated to support V2Planner (POMO-based static model + dynamic adapter).

Public symbols are re-exported here for convenient imports, e.g.:

	from adversarial import build_env, DiffusionAdversarialTrainer, build_planner_optimizer

"""

from .types import EpisodeResult, GeneratorPolicy, PlannerPolicy, DemandTuple
from .builders import (
	build_env, build_planner, build_diffusion,
	get_planner_trainable_params, build_planner_optimizer, save_planner_checkpoint
)
from .co_train.train_coevolution import coevolution_loop, CoevolutionConfig
from .co_train.rl_hooks import reinforce_planner_hook, reinforce_v2_planner_hook
from .co_train.supervised_hooks import supervised_planner_hook

__all__ = [
	"EpisodeResult",
	"GeneratorPolicy",
	"PlannerPolicy",
	"DemandTuple",
	"build_env",
	"build_planner",
	"build_diffusion",
	"get_planner_trainable_params",
	"build_planner_optimizer",
	"save_planner_checkpoint",
	"coevolution_loop",
	"CoevolutionConfig",
	"reinforce_planner_hook",
	"reinforce_v2_planner_hook",
	"supervised_planner_hook",
]
