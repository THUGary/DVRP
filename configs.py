from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Iterator
import itertools


@dataclass
class Config:
	# Environment
	width: int = 20
	height: int = 20
	num_agents: int = 5
	capacity: int = 200
	depot: Tuple[int, int] = (0, 0)
	max_time: int = 100

	# Generator params
	generator_type: str = "net"  # "rule" | "net"
	generator_params: Dict[str, Any] = field(default_factory=lambda: {
		"max_per_step": 2, # not used in rule-based generator
		"total_demand":50,
		"max_time":50, 
		"max_c": 5, # from 1 to 10
		"min_lifetime": 40,
		"max_lifetime": 50,

		"num_centers": 6,
		"distribution": "uniform",  # "uniform" | "gaussian" | "cluster"
		"neighborhood_size": 3, # 3-15, the average radius of the concentrated generation areas
        # add checkpoint path for the network-based generator
		"model_path": "checkpoints/diffusion_model.pth",
	})

	# Planner params
	planner_type: str = "rule"  # "rule" | "net"
	planner_params: Dict[str, Any] = field(default_factory=dict)

	# Controller params
	controller_type: str = "rule"
	controller_params: Dict[str, Any] = field(default_factory=dict)


def get_default_config() -> Config:
	return Config()

# ==============================================================================
# == Parameter Space for Network Generator Data Generation ==
# ==============================================================================
# This defines the universe of parameters for generating the training dataset.
GENERATOR_PARAM_SPACE = {
    "total_demand": [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
    "num_centers": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "distribution": ["uniform", "gaussian", "cluster"],
    "neighborhood_size": [3, 5, 7, 9, 11, 13, 15],
    "max_c": [2, 5, 10],
    "min_lifetime": [30, 60],
    "max_lifetime": [61, 100],
}

def get_param_combinations() -> Iterator[Dict[str, Any]]:
    """
    Creates an iterator that yields all unique and valid combinations of parameters
    defined in GENERATOR_PARAM_SPACE.
    """
    keys = GENERATOR_PARAM_SPACE.keys()
    values = GENERATOR_PARAM_SPACE.values()
    for instance in itertools.product(*values):
        params = dict(zip(keys, instance))
        # Ensure min_lifetime is always less than max_lifetime
        if params.get("min_lifetime", 0) < params.get("max_lifetime", 1):
            yield params

# ==============================================================================
