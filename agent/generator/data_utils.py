"""
Utility functions for preparing and normalizing conditional inputs for the demand generator.
"""
import torch
import numpy as np
from typing import Dict, Any, Optional

from configs import GENERATOR_PARAM_SPACE

# --- Automatically derive Normalization Constants from the Parameter Space ---
CONDITION_NORM = {
    key: float(max(values))
    for key, values in GENERATOR_PARAM_SPACE.items()
    if isinstance(values[0], (int, float))
}

# This must match the dimension of the feature vector created by prepare_condition
CONDITION_DIM = 7

# Default values for condition parameters (used when not specified)
DEFAULT_CONDITION_PARAMS = {
    "total_demand": 60,
    "max_c": 5,
    "num_centers": 6,
    "neighborhood_size": 3,
    "distribution": "gaussian",
}

def normalize_value(val: float, min_val: float, max_val: float) -> float:
    """Normalizes a value from its original range to [-1, 1]."""
    if max_val == min_val:
        return 0.0
    return 2 * ((val - min_val) / (max_val - min_val)) - 1

def unnormalize_value(val: float, min_val: float, max_val: float) -> float:
    """Un-normalizes a value from [-1, 1] to its original range [min_val, max_val]."""
    return (val + 1) / 2 * (max_val - min_val) + min_val

def prepare_condition(
    params: Optional[Dict[str, Any]] = None,
    total_demand: Optional[int] = None,
    max_c: Optional[int] = None,
) -> torch.Tensor:
    """
    Creates the conditional input tensor for the diffusion model.
    
    This function accepts parameters in two ways:
    1. Via params dict with keys like "param_total_demand", "param_max_c", etc.
    2. Via explicit keyword arguments total_demand and max_c
    
    Only total_demand and max_c are configurable; other parameters use fixed defaults.
    
    Args:
        params: Optional dict with "param_*" prefixed keys (legacy support)
        total_demand: Total demand value (overrides params if provided)
        max_c: Max capacity per node (overrides params if provided)
    
    Returns:
        Condition tensor of shape (CONDITION_DIM,)
    """
    if params is None:
        params = {}
    
    # Get total_demand: explicit arg > params > default
    if total_demand is not None:
        _total_demand = total_demand
    elif "param_total_demand" in params:
        _total_demand = params["param_total_demand"]
    elif "total_demand" in params:
        _total_demand = params["total_demand"]
    else:
        _total_demand = DEFAULT_CONDITION_PARAMS["total_demand"]
    
    # Get max_c: explicit arg > params > default
    if max_c is not None:
        _max_c = max_c
    elif "param_max_c" in params:
        _max_c = params["param_max_c"]
    elif "max_c" in params:
        _max_c = params["max_c"]
    else:
        _max_c = DEFAULT_CONDITION_PARAMS["max_c"]
    
    # Use fixed defaults for other parameters
    _num_centers = DEFAULT_CONDITION_PARAMS["num_centers"]
    _neighborhood_size = DEFAULT_CONDITION_PARAMS["neighborhood_size"]
    _distribution = DEFAULT_CONDITION_PARAMS["distribution"]
    
    # Distribution one-hot encoding
    dist_map = {"uniform": 0, "gaussian": 1, "cluster": 2}
    dist_one_hot = np.zeros(3)
    dist_one_hot[dist_map.get(_distribution, 1)] = 1  # Default to gaussian

    # Normalize using constants from config
    total_demand_norm = _total_demand / CONDITION_NORM["total_demand"]
    num_centers_norm = _num_centers / CONDITION_NORM["num_centers"]
    neighborhood_size_norm = _neighborhood_size / CONDITION_NORM["neighborhood_size"]
    max_c_norm = _max_c / CONDITION_NORM["max_c"]

    condition_np = np.array([
        total_demand_norm, num_centers_norm, neighborhood_size_norm, max_c_norm, *dist_one_hot
    ])
    
    return torch.from_numpy(condition_np).float()