"""
Utility functions for preparing and normalizing conditional inputs for the demand generator.
"""
import torch
import numpy as np
from typing import Dict, Any

from configs import GENERATOR_PARAM_SPACE

# --- Automatically derive Normalization Constants from the Parameter Space ---
CONDITION_NORM = {
    key: float(max(values))
    for key, values in GENERATOR_PARAM_SPACE.items()
    if isinstance(values[0], (int, float))
}

# This must match the dimension of the feature vector created by prepare_condition
CONDITION_DIM = 7 

def normalize_value(val: float, min_val: float, max_val: float) -> float:
    """Normalizes a value from its original range to [-1, 1]."""
    if max_val == min_val:
        return 0.0
    return 2 * ((val - min_val) / (max_val - min_val)) - 1

def unnormalize_value(val: float, min_val: float, max_val: float) -> float:
    """Un-normalizes a value from [-1, 1] to its original range [min_val, max_val]."""
    return (val + 1) / 2 * (max_val - min_val) + min_val

def prepare_condition(params: Dict[str, Any]) -> torch.Tensor:
    """
    Creates the conditional input tensor from a dictionary of parameters.
    This is the single source of truth for creating condition vectors.
    """
    dist_map = {"uniform": 0, "gaussian": 1, "cluster": 2}
    dist_type = params.get("param_distribution", "uniform")
    dist_one_hot = np.zeros(3)
    dist_one_hot[dist_map.get(dist_type, 0)] = 1

    # Normalization now uses the robust constants from the top of this file
    total_demand = params.get("param_total_demand", 50) / CONDITION_NORM["total_demand"]
    num_centers = params.get("param_num_centers", 6) / CONDITION_NORM["num_centers"]
    neighborhood_size = params.get("param_neighborhood_size", 3) / CONDITION_NORM["neighborhood_size"]
    max_c = params.get("param_max_c", 5) / CONDITION_NORM["max_c"]

    condition_np = np.array([
        total_demand, num_centers, neighborhood_size, max_c, *dist_one_hot
    ])
    
    return torch.from_numpy(condition_np).float()