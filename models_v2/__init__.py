# Simplified VRP Model Architecture
# - StaticVRPModel: Pure POMO-style model for static VRP (multi-vehicle)
# - DynamicVRPModel: Static model + residual adapter for DVRP

from .static_model import StaticVRPModel, StaticVRPEnv
from .dynamic_model import DynamicVRPModel

__all__ = ["StaticVRPModel", "StaticVRPEnv", "DynamicVRPModel"]
