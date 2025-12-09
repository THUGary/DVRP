# ...existing code...
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Iterator, Optional
import itertools

# ==============================================================================
# == Normalization Constants (MUST match training configuration) ==
# ==============================================================================
# These constants define the normalization scheme used during BOTH training and inference.
# NOTE: Changing these requires retraining the model from scratch!
#
# NORMALIZATION SCHEME (v2 - capacity-normalized):
# ------------------------------------------------
# 1. Coordinates: coord_norm = coord / COORD_NORM (maps to [0,1] range)
# 2. Demands: demand_norm = demand / DEMAND_NORM (maps demands to model scale)
# 3. Vehicle Capacity: vehicle_capacity = capacity / DEMAND_NORM
#    - With DEMAND_NORM=30 and capacity=30, vehicle_capacity=1.0 (full capacity)
#    - This makes capacity=30 the "unit" or reference capacity
#
# KEY DESIGN DECISIONS:
# ---------------------
# - Vehicle capacity is FIXED at 30 for all training and inference
# - Max demand per node is FIXED at 5 (so max 6 nodes can fill one vehicle)
# - Map size (COORD_NORM) can be varied: 20, 30, 40, etc.
# - Total demand count can be varied: 20, 30, 50, etc.
#
# TRAINING CONFIGURATION (Static VRP Model):
# - Map: [0,1] x [0,1] (unit square, scaled by COORD_NORM)
# - Vehicle capacity: 1.0 (represents capacity=30 in real terms)
# - Demands: random [1,5] / 30, so each demand is 0.033-0.167 of capacity
#
# INFERENCE CONFIGURATION:
# - Map: [0, COORD_NORM] x [0, COORD_NORM] (integer grid)
# - Vehicle capacity: 30 (fixed, model sees 30/30=1.0)
# - Demands: [1, max_c] where max_c=5 (fixed)

# Map size - can be changed for different problem scales
# NOTE: Model is trained on normalized [0,1] space, so larger maps
#       should work as long as COORD_NORM is updated to match
COORD_NORM: float = 20.0  # Default grid size (can be 20, 30, 40, etc.)

# FIXED: Demand normalization = vehicle capacity (makes capacity=1.0 in model)
# This is the key normalization constant - DO NOT CHANGE after training!
DEMAND_NORM: float = 30.0  # = vehicle capacity (model sees demands/30, capacity=30/30=1.0)

# Backward compatibility alias (deprecated, use DEMAND_NORM instead)
DEMAND_NORM_TRAINING: float = DEMAND_NORM

# FIXED: Vehicle capacity and max demand per node
# These define the problem structure and should NOT be changed
DEFAULT_CAPACITY: int = 30  # Fixed vehicle capacity (model sees 30/30=1.0)
DEFAULT_MAX_DEMAND: int = 5  # Fixed max demand per node (model sees 5/30=0.167)

# ==============================================================================


@dataclass
class Config:
    # Environment
    width: int = 20
    height: int = 20
    num_agents: int = 2
    capacity: int = DEFAULT_CAPACITY  # Use centralized default (30)
    depot: Tuple[int, int] = (0, 0)
    max_time: int = 100 # the value has to be consistent with generator_params' max_time
    # Hard cap on episode after last generation time; if None, will be set in __post_init__
    max_end_time: Optional[int] = None
    include_service_time: bool = False #Attention:  Overwritten in run.py and benchmark.py
    # Reward scales
    # Reward scale tweaks tuned for smaller agent counts (e.g., 2 vehicles)
    capacity_reward_scale: float = 0.25
    expiry_penalty_scale: float = 0.02      # 从 0.05 降一点
    switch_penalty_scale: float = 0.0       # RL reward now omits direction-change penalty
    # Density-scaled pairwise distance penalty between agents
    distance_penalty_base: float = 0.0001
    
    distance_penalty_min_dist: float = 1.5
    move_penalty_scale: float = 0.02
    depot_return_bonus_scale: float = 0.05
    approach_bonus_scale: float = 0.02
    approach_bonus_max_dist: float = 6.0
    # Per-step waiting penalty over active (unserved) demands
    wait_penalty_scale: float = 0.0003
    # Exploration penalty params
    exploration_history_n: int = 3
    exploration_penalty_scale: float = 0.0  # 先关掉探索惩罚

    # Generator params
    # Use "rule" for static VRP evaluation, "net" for dynamic with diffusion model
    generator_type: str = "rule"  # "rule" | "net"
    generator_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_per_step": 2, # not used in rule-based generator
        "depot": "__depot__",  # placeholder to be replaced with Config.depot (accepts "__depot__" or "__DEPOT__")
        "max_time": "__MAX_TIME__",  # placeholder to be replaced with Config.max_time
        # Limiting modes (num_nodes takes priority if set):
        # - num_nodes: limit by number of demand nodes (preferred)
        # - total_demand: limit by sum of all demand capacities (legacy)
        "num_nodes": 20,  # Number of demand nodes to generate
        "total_demand": 60,  # Legacy: upper limit of sum of all demands (used if num_nodes not set)
        "max_c": 5, # from 1 to 10
        "min_lifetime": 40,
        "max_lifetime": 50,
        "min_service_time": 1,
        "max_service_time": 3,
        "service_time_per_unit": 0.0,

        "num_centers": 6,
        "distribution": "gaussian",  # "uniform" | "gaussian" | "cluster"
        "neighborhood_size": 5, # 3-15, the average radius of the concentrated generation areas
        "burst_prob": 0.1, # 0.0 - 1.0, probability of bursting demands among all demands
        # Checkpoint path for the network-based generator (only used when generator_type="net")
        "model_path": "checkpoints/diffusion_model.pth",
    })

    # Planner params
    planner_type: str = "rule"  # "rule" | "net" (ATTENTION: this is overwritten in run.py based on args)
    planner_params: Dict[str, Any] = field(default_factory=dict)

    prompt_planner_params: Dict[str, Any] = field(default_factory=lambda: {
        "model_path": "checkpoints/prompt_vrp/checkpoint-10000.pt",
        "keys_path": "models_v2/keys_new_16",
        "augmentation": True,
    })
    
    # V2Planner params (POMO-based architecture)
    v2_planner_params: Dict[str, Any] = field(default_factory=lambda: {
        "static_ckpt": "checkpoints/static_vrp_v2/best_n20.pt",
        "adapter_ckpt": "checkpoints/dynamic_adapter_v2/best_adapter.pt",
        "device": "cuda",
        "pomo_size": 20,
        "augmentation": True,
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
