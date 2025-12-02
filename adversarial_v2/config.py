"""
JUST dataclass defination

Configuration dataclasses for adversarial co-evolution training.

TERMINOLOGY (consistent with other scripts):
- map_size: Side length of the square map 
- total_demand: Upper limit of the sum of all customer demands (total capacity to serve)
- max_c: Maximum demand per node (default 5, so demands are 1-5)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvironmentConfig:
    """
    Configuration for environment (shared between planner and generator).
    """
    # Grid settings (square map: map_size × map_size)
    map_size: int = 30
    
    # Agent settings
    num_agents: int = 5
    capacity: int = 30  # Fixed vehicle capacity (= DEMAND_NORM)
    
    # Time settings  
    max_time: int = 100
    max_end_time: int = 200
    
    # Depot (can be randomized)
    depot: tuple = (0, 0)
    randomize_depot: bool = True
    
    # Demand generation settings
    num_nodes: int = 50  # upper limit of number of demand nodes
    total_demand: int = 150  # Upper limit of sum of all demands (~num_nodes * avg_demand)
    max_c: int = 5  # Max demand per node (demands are 1 to max_c)
    min_lifetime: int = 10
    max_lifetime: int = 50


@dataclass
class CoevolutionConfig:
    """
    Main configuration for co-evolution training.
    
    Model-specific configs are NOT duplicated here. They use defaults from:
    - Planner: training_v2/train_static.py (embedding_dim=128, encoder_layers=6, etc.)
    - Generator: training/generator/adversarial_trainer.py (AdvConfig: lr=1e-4, baseline_beta=0.9, etc.)
    """
    # Training mode: "static" or "dynamic"
    # - static: All demands appear at t=0 with deadline=max_time
    # - dynamic: Demands appear at different times with various deadlines
    mode: str = "static"
    
    # Co-evolution cycles
    num_cycles: int = 10
    planner_epochs_per_cycle: int = 5
    generator_epochs_per_cycle: int = 5
    
    # First cycle can have different epochs for longer initial training
    # If None, uses planner_epochs_per_cycle
    first_cycle_planner_epochs: Optional[int] = None
    
    # Batch settings (shared)
    # Memory usage ≈ batch_size × pomo_size × num_nodes^2 × embedding_dim × 4 bytes
    batch_size: int = 64  # Reduce for limited GPU memory
    pomo_size: int = 100   # POMO parallel rollouts (reduce for limited GPU memory)
    
    episodes_per_epoch: int = 1000
    
    # Version sampling policy for planner training
        # "uniform": sample uniformly from all versions
        # "latest_biased": bias towards recent versions
        # "all": use all versions in each epoch
    version_sample_policy: str = "latest_biased"
    latest_bias: float = 0.7  # P(sample latest) when latest_biased
    
    # Hardware
    device: str = "cuda"
    seed: int = 42
    
    # Checkpointing
    save_dir: str = "checkpoints/adversarial_v2"
    save_interval: int = 1  # Save every N cycles
    
    # Checkpoint paths (optional, for loading pretrained models)
    planner_checkpoint: Optional[str] = None
    generator_checkpoint: Optional[str] = None
    
    # Environment config
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.mode in ("static", "dynamic"), f"Mode must be 'static' or 'dynamic', got {self.mode}"
        assert self.version_sample_policy in ("uniform", "latest_biased", "all"), \
            f"Invalid version_sample_policy: {self.version_sample_policy}"
        # Ensure capacity matches DEMAND_NORM (fixed at 30)
        assert self.env.capacity == 30, f"Vehicle capacity must be 30 (DEMAND_NORM), got {self.env.capacity}"
