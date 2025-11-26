# ...existing code...
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Iterator, Optional
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
    # Hard cap on episode after last generation time; if None, will be set in __post_init__
    max_end_time: Optional[int] = None
    include_service_time: bool = False
    # Reward scales
    # Reward scale tweaks tuned for smaller agent counts (e.g., 2 vehicles)
    capacity_reward_scale: float = 0.25
    expiry_penalty_scale: float = 0.02      # 从 0.05 降一点
    switch_penalty_scale: float = 0.0       # RL reward now omits direction-change penalty
    # Density-scaled pairwise distance penalty between agents
    distance_penalty_base: float = 0.0001
    
    distance_penalty_min_dist: float = 1.5
    move_penalty_scale: float = 0.0002
    approach_bonus_scale: float = 0.02
    approach_bonus_max_dist: float = 6.0
    # Per-step waiting penalty over active (unserved) demands
    wait_penalty_scale: float = 0.0003
    # Exploration penalty params
    exploration_history_n: int = 3
    exploration_penalty_scale: float = 0.0  # 先关掉探索惩罚

    # Generator params
    generator_type: str = "net"  # "rule" | "net"
    generator_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_per_step": 2, # not used in rule-based generator
        "depot": "__depot__",  # placeholder to be replaced with Config.depot (accepts "__depot__" or "__DEPOT__")
        "max_time": "__MAX_TIME__",  # placeholder to be replaced with Config.max_time
        "total_demand":20,
        "max_c": 10, # from 1 to 10
        "min_lifetime": 40,
        "max_lifetime": 50,
        "min_service_time": 1,
        "max_service_time": 3,
        "service_time_per_unit": 0.0,

        "num_centers": 6,
        "distribution": "uniform",  # "uniform" | "gaussian" | "cluster"
        "neighborhood_size": 3, # 3-15, the average radius of the concentrated generation areas
        "burst_prob": 0.1, # 0.0 - 1.0, probability of bursting demands among all demands
        # add checkpoint path for the network-based generator
        "model_path": "checkpoints/rl_generator/greedy_20251126-120703/ckpt_ep_2300.pth",#"checkpoints/diffusion_model.pth",
        # "model_path": "checkpoints/diffusion_model.pth",
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
        "coord_norm": 20.0,
        "capacity_norm": 200.0,
        "time_norm": 100.0,
        "adapter_dim": 0,
        "ckpt": "checkpoints/planner/planner_dynamic_20_2_200.pt",
    })
    cvrp_planner_params: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "pomo_root": "~/POMO-master/NEW_py_ver/CVRP/POMO",
        "checkpoint": None,
        "device": "cpu",
        "max_nodes": 100,
        "coord_normalizer": None,  # if None use max(width, height)
        "env_params": {
            "problem_size": 100,
            "pomo_size": 100,
        },
        "model_params": {
            "embedding_dim": 128,
            "sqrt_embedding_dim": 128 ** 0.5,
            "encoder_layer_num": 6,
            "qkv_dim": 16,
            "head_num": 8,
            "logit_clipping": 10,
            "ff_hidden_dim": 512,
            "eval_type": "argmax",
        },
        # when more demands than max_nodes, keep top-k by earliest deadline
        "selection_policy": "earliest_due",
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
        # ensure service-time range is well-ordered
        min_service = int(self.generator_params.get("min_service_time", 0))
        max_service = int(self.generator_params.get("max_service_time", min_service))
        if max_service < min_service:
            max_service = min_service
        self.generator_params["min_service_time"] = min_service
        self.generator_params["max_service_time"] = max_service
        # default max_end_time if not provided: allow time after last generation
        # to return to depot and finish remaining work
        if self.max_end_time is None:
            # heuristic default: 2x max_time to allow wind-down
            self.max_end_time = int(self.max_time * 2)


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
