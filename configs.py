# ...existing code...
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
    max_time: int = 100 # the value has to be consistent with generator_params' max_time
    # Reward scales
    capacity_reward_scale: float = 0.05
    expiry_penalty_scale: float = 0.05
    switch_penalty_scale: float = 0.001
    # Exploration penalty params
    exploration_history_n: int = 3  # consider positions at t-2 .. t-n
    exploration_penalty_scale: float = 0.001  # scale of revisit penalty

    # Generator params
    generator_type: str = "rule"  # "rule" | "net"
    generator_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_per_step": 2, # not used in rule-based generator
        "depot": "__depot__",  # placeholder to be replaced with Config.depot (accepts "__depot__" or "__DEPOT__")
        "max_time": "__MAX_TIME__",  # placeholder to be replaced with Config.max_time
        "total_demand":50,
        "max_c": 5, # from 1 to 10
        "min_lifetime": 40,
        "max_lifetime": 50,

        "num_centers": 6,
        "distribution": "uniform",  # "uniform" | "gaussian" | "cluster"
        "neighborhood_size": 3, # 3-15, the average radius of the concentrated generation areas
        "burst_prob": 0.1, # 0.0 - 1.0, probability of bursting demands among all demands
        # add checkpoint path for the network-based generator
    "model_path": "checkpoints/coevolution/generator_cycle_4.pth",#"checkpoints/diffusion_model.pth",
    })

    # Planner params
    planner_type: str = "rule"  # "rule" | "net"
    planner_params: Dict[str, Any] = field(default_factory=dict)
    model_planner_params: Dict[str, Any] = field(default_factory=lambda: {
        "time_plan": 3,
        "device": "cpu",
        "lateness_lambda": 0.0,
        "d_model": 128,
        "nhead": 8,
        "nlayers": 2,
        # "ckpt": "checkpoints/planner/planner_20_2_10.pt",
        "ckpt": "training/planner/planner_rl_best.pt",
    })

    # Controller params
    controller_type: str = "rule"
    controller_params: Dict[str, Any] = field(default_factory=dict)
      
    def __post_init__(self):
        # normalize max_time placeholder
        if self.generator_params.get("max_time") == "__MAX_TIME__":
            self.generator_params["max_time"] = self.max_time
        # accept either "__depot__" or "__DEPOT__" as placeholder
        if self.generator_params.get("depot") in ("__depot__", "__DEPOT__"):
            self.generator_params["depot"] = self.depot


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
# ...existing code...