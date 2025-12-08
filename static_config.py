# ==============================================================================
# Static Configuration File for DVRP Project
# ==============================================================================
# TODO: 每个参数表明使用处：cotraim, regular_train, evaluate, show_diffusion

from typing import Tuple

STATIC_DEMANDS: bool = True       # Use static demands mode (evaluate_distributions.sh, run_dvrp.sh)
DEVICE: str = "cuda"              # Device: "cuda" or "cpu" (run_cotrain.sh, train_static.sh)

# ==============================================================================
# 1. NORMALIZATION CONSTANTS (FIXED - DO NOT CHANGE)
# ==============================================================================
# These constants are fixed during model training. Changing them requires retraining!
# Source: configs.py

COORD_NORM: float = 20.0          # Map coordinate normalization factor (configs.py)
DEMAND_NORM: float = 30.0         # Demand normalization = vehicle capacity (configs.py)
DEFAULT_CAPACITY: int = 30        # Fixed vehicle capacity (configs.py, run_cotrain.sh)
DEFAULT_MAX_DEMAND: int = 5       # Fixed max demand per node (configs.py)


# ==============================================================================
# 2. ENVIRONMENT SETTINGS
# ==============================================================================
# Basic environment configuration for VRP simulation

NUM_AGENTS: int = 2               # Number of vehicles (evaluate_distributions.sh, run_cotrain.sh, run_dvrp.sh, configs.py)
MAP_SIZE: int = 30                # Square map side length (evaluate_distributions.sh, run_cotrain.sh, run_dvrp.sh, show_diffusion.sh, train_static.sh)
DEPOT: Tuple[int, int] = (0, 0)   # Depot location (configs.py)
CAPACITY: int = 30                # Vehicle capacity (evaluate_distributions.sh, run_cotrain.sh, configs.py)


# ==============================================================================
# 3. DEMAND SETTINGS
# ==============================================================================
# Parameters controlling demand generation

NUM_NODES: int = 20               # Number of demand nodes (evaluate_distributions.sh, run_cotrain.sh, run_dvrp.sh, show_diffusion.sh, train_static.sh, configs.py)
TOTAL_DEMAND: int = 60            # Upper limit of sum of all demands (evaluate_distributions.sh, run_cotrain.sh, run_dvrp.sh, train_static.sh, configs.py)
MAX_C: int = 5                    # Max demand per node, 1 to max_c (evaluate_distributions.sh, run_cotrain.sh, run_dvrp.sh, configs.py)
MIN_LIFETIME: int = 10            # Min demand lifetime (run_cotrain.sh, configs.py)
MAX_LIFETIME: int = 50            # Max demand lifetime (run_cotrain.sh, configs.py)
RANDOMIZE_DEPOT: bool = True      # Randomize depot location in cotrain (run_cotrain.sh)


# ==============================================================================
# 4. TIME SETTINGS
# ==============================================================================
# Episode time limits and step constraints

MAX_TIME: int = 5000              # Max simulation time / steps (all scripts)


# ==============================================================================
# 5. EVALUATION SETTINGS
# ==============================================================================
# Parameters for model evaluation

NUM_RUNS: int = 20                # Number of evaluation runs per distribution (evaluate_distributions.sh)
SEED: int = 42                    # Random seed (all scripts)


# ==============================================================================
# 6. POMO INFERENCE PARAMETERS
# ==============================================================================
# Parameters for POMO-based model inference

POMO_SIZE: int = 100              # Number of parallel rollouts (evaluate_distributions.sh, run_cotrain.sh, configs.py)
AUG_FACTOR: int = 8               # Data augmentation factor (evaluate_distributions.sh, train_static.sh)


# ==============================================================================
# 7. PLANNER SETTINGS
# ==============================================================================
# Rule-based and model-based planner configuration

# Rule-based planner modes (evaluate_distributions.sh)
RULE_MODES: str = "optimize,greedy,heuristic"

# TODO: ??
# Global optimization modes (evaluate_distributions.sh)
GLOBAL_OPT_MODES: str = ""

# TODO: 不同文件使用不同参数
# Model checkpoint paths
MODEL_CHECKPOINTS: str = "checkpoints/cotrain/static_20251205_144853/planner_cycle_1_best.pt"  # (evaluate_distributions.sh)
STATIC_CKPT: str = "checkpoints/cotrain/static_20251207_131359/planner_cycle_1_best.pt"        # (run_dvrp.sh)
RULE_MODE: str = ""               # Rule-based planner mode: "greedy", "exact", "heuristic" (run_dvrp.sh)


# ==============================================================================
# 8. GENERATOR SETTINGS
# ==============================================================================
# Diffusion generator configuration

GENERATOR_TYPE: str = "net"       # Generator type: "rule" or "net" (run_dvrp.sh, configs.py)

# Diffusion model checkpoints (evaluate_distributions.sh, run_dvrp.sh, show_diffusion.sh)
# TODO: 明确变量名，run、show_diffusion使用单个ckpt，evaluate使用多个ckpt
DIFFUSION_CHECKPOINTS: str = (
    "version0=checkpoints/cotrain/static_20251205_144853/generator_v0.pth,"
    "version1=checkpoints/cotrain/static_20251205_144853/generator_cycle_1.pth,"
    "version2=checkpoints/cotrain/static_20251205_144853/generator_cycle_2.pth,"
    "version3=checkpoints/cotrain/static_20251205_144853/generator_cycle_3.pth,"
    "version4=checkpoints/cotrain/static_20251205_144853/generator_cycle_4.pth,"
    "version5=checkpoints/cotrain/static_20251205_144853/generator_cycle_5.pth"
)
DIFFUSION_CKPT: str = "checkpoints/cotrain/static_20251207_131359/generator_cycle_5.pth"  # (run_dvrp.sh)
SHOW_DIFFUSION_CHECKPOINT: str = "checkpoints/cotrain/static_20251205_065614/generator_cycle_1.pth"  # (show_diffusion.sh)

# Generator model path in configs (configs.py)：run.py
GENERATOR_MODEL_PATH: str = "checkpoints/rl_generator/greedy_static_20251205-142015/best.pth"


# ==============================================================================
# 9. DIFFUSION PROBLEM BANK (Optional Two-Stage Workflow)
# ==============================================================================
# For pregenerated diffusion problems (evaluate_distributions.sh)

PROBLEM_BANK_IN: str = ""         # Existing JSON file to read pregenerated problems
PROBLEM_BANK_OUT: str = ""        # Output JSON file to store generated problems
GENERATE_ONLY: bool = False       # Only generate problems without running planners


# ==============================================================================
# 10. CO-EVOLUTION TRAINING SETTINGS
# ==============================================================================
# Settings for adversarial co-training (run_cotrain.sh)

# TODO： static 合并
MODE: str = "static"              # Training mode: "static" or "dynamic" (run_cotrain.sh)
NUM_GPUS: int = 1                 # Number of GPUs for DDP training (run_cotrain.sh)
NUM_CYCLES: int = 2              # Number of co-evolution cycles (run_cotrain.sh)
FIRST_CYCLE_PLANNER_EPOCHS: int = 1  # First cycle planner epochs (run_cotrain.sh)
PLANNER_EPOCHS: int = 1          # Planner training epochs per cycle (run_cotrain.sh)
GENERATOR_EPOCHS: int = 1       # Generator training epochs per cycle (run_cotrain.sh)


# ==============================================================================
# 11. EARLY STOPPING SETTINGS
# ==============================================================================
# Early stopping configuration for training (run_cotrain.sh, train_static.sh)

# Planner early stopping (run_cotrain.sh)
PLANNER_EARLY_STOP_PATIENCE: int = 20    # Epochs without improvement to trigger stop
PLANNER_EARLY_STOP_THRESHOLD: float = 0.1  # Minimum improvement threshold

# Generator early stopping (run_cotrain.sh)
GENERATOR_EARLY_STOP_PATIENCE: int = 20  # Epochs without improvement to trigger stop
GENERATOR_EARLY_STOP_THRESHOLD: float = 1.0  # Minimum improvement threshold

# Static training early stopping (train_static.sh)
PATIENCE: int = 50                # Early stopping patience
THRESHOLD: float = 0.0001         # Early stopping improvement threshold


# ==============================================================================
# 12. BATCH AND TRAINING SETTINGS
# ==============================================================================
# Batch sizes and training hyperparameters

BATCH_SIZE: int = 1             # Batch size for training (run_cotrain.sh)
EPISODES_PER_EPOCH: int = 1  # Episodes per epoch (run_cotrain.sh)

TRAIN_EPISODES_PER_EPOCH: int = 10000  # Episodes per epoch for static training (train_static.sh)
TRAIN_BATCH_SIZE: int = 4        # Batch size for static training (train_static.sh)
LR: str = "1e-4"                  # Learning rate (train_static.sh)
EPOCHS: int = 2000                # Training epochs (train_static.sh)


# ==============================================================================
# 13. PROBLEM CACHE SETTINGS
# ==============================================================================
# Memory management for problem caching (run_cotrain.sh)

CACHE_REUSE_RATIO: float = 0.7    # Probability of using cached problems (run_cotrain.sh)
MAX_PROBLEMS_PER_VERSION: int = 30000  # Max problems to cache per version (run_cotrain.sh)
MIN_CACHE_SIZE_FOR_REUSE: int = 5000   # Min cache size before enabling reuse (run_cotrain.sh)


# ==============================================================================
# 14. VERSION SAMPLING SETTINGS
# ==============================================================================
# To prevent policy cycling in co-evolution (run_cotrain.sh)

VERSION_POLICY: str = "uniform"   # Options: "uniform", "latest_biased", "all" (run_cotrain.sh)
LATEST_BIAS: float = 0.3          # P(sample latest) when latest_biased (run_cotrain.sh)


# ==============================================================================
# 15. MODEL ARCHITECTURE SETTINGS
# ==============================================================================
# Neural network architecture configuration (train_static.sh)

# TODO： 合并 TARGET_VEHICLES ??
EMBEDDING_DIM: int = 128          # Embedding dimension (train_static.sh)
ENCODER_LAYERS: int = 6           # Number of encoder layers (train_static.sh)
HEADS: int = 8                    # Number of attention heads (train_static.sh)
TARGET_VEHICLES: int = 2          # Target number of vehicles (train_static.sh)


# ==============================================================================
# 15b. PRE-GENERATED DATASET SETTINGS
# ==============================================================================
# Settings for pre-generated training/test datasets (generate_dataset.py, train_static.sh)

# Dataset paths
TRAIN_DATA_PATH: str = "data/static_diffusion_vrp/train.pt"         # Path to pre-generated training data (.pt file), empty = random generation
TEST_DATA_PATH: str = "data/static_diffusion_vrp/test.pt"          # Path to pre-generated test data (.pt file)

# Dataset generation settings (generate_dataset.py)
DATASET_OUTPUT_DIR: str = "data/static_diffusion_vrp"  # Output directory for generated datasets
DATASET_TOTAL_EPISODES: int = int(3e6)          # Total number of problems to generate
DATASET_TEST_RATIO: float = 0.1              # Ratio of problems for test set (0.0 to 1.0)
DATASET_MODE: str = "diffusion"                 # Generation mode: "random" or "diffusion"
DATASET_DIFFUSION_CKPT: str = "checkpoints/rl_generator/greedy_20251207-181737/best.pth"             # Diffusion model checkpoint for dataset generation
DATASET_DDIM_STEPS: int = 50                 # DDIM sampling steps for diffusion generation
DATASET_BATCH_SIZE: int = 64                # Batch size for diffusion generation


# ==============================================================================
# 16. DIFFUSION VISUALIZATION SETTINGS
# ==============================================================================
# Parameters for diffusion model visualization (show_diffusion.sh)

VIZ_MODE: str = "heatmap"         # Visualization mode: "heatmap", "episode", "compare" (show_diffusion.sh)
NUM_SAMPLES: int = 50             # Number of sampling iterations (show_diffusion.sh)
VIZ_SAVE_PATH: str = "outputs/heatmap1.png"  # Output save path (show_diffusion.sh)


# ==============================================================================
# 17. OUTPUT AND CHECKPOINT SETTINGS
# ==============================================================================
# Paths for saving outputs and checkpoints

OUT_DIR: str = "outputs/eval"     # Evaluation output directory (evaluate_distributions.sh)
SAVE_DIR: str = "checkpoints/static_vrp_v2"  # Checkpoint save directory (train_static.sh)
SAVE_INTERVAL: int = 10           # Save checkpoint every N epochs (train_static.sh)
PLOT_METRICS: str = "failure_rate,total_distance,inference_time_total"  # Metrics to plot (evaluate_distributions.sh)

# Resume training checkpoint (train_static.sh)
RESUME_FROM: str = ""             # Path to resume from (empty = train from scratch)

# Co-train checkpoints (run_cotrain.sh)
PLANNER_INITIALIZE: str = "checkpoints/static_vrp_v2/best_n20.pt"  # Pretrained planner path
GENERATOR_INITIALIZE: str = "checkpoints/rl_generator/greedy_static_20251205-142015/best.pth"  # Pretrained generator path
COTRAIN_RESUME_FROM: str = ""     # Resume from checkpoint directory


# ==============================================================================
# 18. RENDERING AND UI SETTINGS
# ==============================================================================
# Visualization and rendering options (run_dvrp.sh)

RENDER: bool = True               # Enable pygame rendering (run_dvrp.sh)
SAVE_RUN: bool = True             # Save run outputs (run_dvrp.sh)


# ==============================================================================
# 19. V2 PLANNER SETTINGS
# ==============================================================================
# POMO-based V2Planner configuration (configs.py)

# TODO: v2是什么 ？？
V2_STATIC_CKPT: str = "checkpoints/static_vrp_v2/best_n20.pt"  # Static model checkpoint
V2_ADAPTER_CKPT: str = "checkpoints/dynamic_adapter_v2/best_adapter.pt"  # Adapter checkpoint
V2_POMO_SIZE: int = 20            # POMO parallel rollouts for V2Planner
V2_AUGMENTATION: bool = True      # Enable data augmentation


# ==============================================================================
# 20. REWARD SCALE SETTINGS
# ==============================================================================
# Reward function configuration (configs.py)

# TODO： 删除不用的
CAPACITY_REWARD_SCALE: float = 0.25      # Capacity reward scale
EXPIRY_PENALTY_SCALE: float = 0.02       # Expiry penalty scale
SWITCH_PENALTY_SCALE: float = 0.0        # Direction switch penalty (disabled)
DISTANCE_PENALTY_BASE: float = 0.0001    # Pairwise distance penalty base
DISTANCE_PENALTY_MIN_DIST: float = 1.5   # Min distance for penalty
MOVE_PENALTY_SCALE: float = 0.02         # Move penalty scale
DEPOT_RETURN_BONUS_SCALE: float = 0.05   # Depot return bonus scale
APPROACH_BONUS_SCALE: float = 0.02       # Approach bonus scale
APPROACH_BONUS_MAX_DIST: float = 6.0     # Max distance for approach bonus
WAIT_PENALTY_SCALE: float = 0.0003       # Per-step waiting penalty
EXPLORATION_HISTORY_N: int = 3           # Exploration history length
EXPLORATION_PENALTY_SCALE: float = 0.0   # Exploration penalty (disabled)


# ==============================================================================
# 21. SERVICE TIME SETTINGS
# ==============================================================================
# Service time configuration for demands (configs.py)

# TODO： min_service_time 前面有吗
INCLUDE_SERVICE_TIME: bool = False       # Include service time in simulation
MIN_SERVICE_TIME: int = 1                # Minimum service time
MAX_SERVICE_TIME: int = 3                # Maximum service time
SERVICE_TIME_PER_UNIT: float = 0.0       # Service time per demand unit


# ==============================================================================
# 22. GENERATOR DISTRIBUTION SETTINGS
# ==============================================================================
# Spatial distribution configuration for demand generation (configs.py)

NUM_CENTERS: int = 6              # Number of cluster centers
DISTRIBUTION: str = "gaussian"    # Distribution type: "uniform", "gaussian", "cluster"
NEIGHBORHOOD_SIZE: int = 5        # Average radius of concentrated areas (3-15)
BURST_PROB: float = 0.1           # Probability of burst demands (0.0-1.0)
MAX_PER_STEP: int = 2             # Max demands generated per step


# ==============================================================================
# 23. GENERATOR PARAMETER SPACE (For Training Dataset)
# ==============================================================================
# Parameter space for generating training data (configs.py)

# TODO： 有实际使用吗，没有的话，注释掉
GENERATOR_PARAM_SPACE = {
    "total_demand": [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
    "num_centers": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "distribution": ["uniform", "gaussian", "cluster"],
    "neighborhood_size": [3, 5, 7, 9, 11, 13, 15],
    "max_c": [2, 5, 10],
    "min_lifetime": [30, 60],
    "max_lifetime": [61, 100],
}


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_config_summary() -> str:
    """Return a formatted summary of key configuration values."""
    return f"""
        ========================================
        DVRP Static Configuration Summary
        ========================================

        Environment:
        - Map size: {MAP_SIZE}x{MAP_SIZE}
        - Num agents: {NUM_AGENTS}
        - Capacity: {CAPACITY}

        Demands:
        - Num nodes: {NUM_NODES}
        - Total demand: {TOTAL_DEMAND}
        - Max demand/node: {MAX_C}

        Training:
        - Mode: {MODE}
        - Epochs: {EPOCHS}
        - Batch size: {BATCH_SIZE}
        - POMO size: {POMO_SIZE}

        Device: {DEVICE}
        ========================================
    """


if __name__ == "__main__":
    print(get_config_summary())
