#!/bin/bash
# =============================================================================
# Co-evolution training script for V2Planner and Diffusion Generator
# Usage: bash scripts/run_cotrain.sh
#
# Edit the configuration variables below to change settings.
#
# Multi-GPU Training:
#   Set NUM_GPUS > 1 to enable DDP training with torchrun
# =============================================================================

set -e

# Go to project root
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

# =============================================================================
# CONFIGURATION - Edit these variables to change settings
# =============================================================================

# --- Training Mode ---
MODE="static"                    # "static" or "dynamic"

# --- Multi-GPU Settings ---
NUM_GPUS=4                       # Number of GPUs (1=single GPU, >1=DDP with torchrun)

# --- Co-evolution Settings ---
NUM_CYCLES=10             # Number of co-evolution cycles
PLANNER_EPOCHS=50        # Planner training epochs per cycle
FIRST_CYCLE_PLANNER_EPOCHS=400  # First cycle planner epochs (leave empty to use PLANNER_EPOCHS)
GENERATOR_EPOCHS=50      # Generator training epochs per cycle

# --- Planner Early Stopping (within each cycle) ---
# Stop planner training early if score doesn't improve for PATIENCE epochs
# Set to 0 or empty to disable early stopping
PLANNER_EARLY_STOP_PATIENCE=20    # Number of epochs without improvement to trigger early stop
PLANNER_EARLY_STOP_THRESHOLD=0.1  # Minimum improvement threshold

# --- Generator Early Stopping (within each cycle) ---
# Stop generator training early if gen_reward doesn't improve for PATIENCE epochs
# Set to 0 or empty to disable early stopping
GENERATOR_EARLY_STOP_PATIENCE=20   # Number of epochs without improvement to trigger early stop (0 = disabled)
GENERATOR_EARLY_STOP_THRESHOLD=1  # Minimum improvement threshold for gen_reward

# --- Batch Settings ---
BATCH_SIZE=512            # Batch size for training
POMO_SIZE=100            # POMO parallel rollouts
EPISODES_PER_EPOCH=10240   # Episodes per epoch

# --- Problem Cache Settings ---
# Memory usage: ~0.6 MB per 1000 problems per version (num_nodes=50)
CACHE_REUSE_RATIO=0.7        # Probability of using cached problems (0.0=always generate, 1.0=always cache)
MAX_PROBLEMS_PER_VERSION=30000 # Max problems to cache per generator version
MIN_CACHE_SIZE_FOR_REUSE=5000  # Minimum cache size before enabling reuse

# --- Version Sampling (to prevent policy cycling) ---
VERSION_POLICY="uniform"  # "uniform", "latest_biased", "all"
LATEST_BIAS=0.3          # P(sample latest) when latest_biased

# --- Environment Settings ---
# map_size: Side length of the square map (map is map_size × map_size)
MAP_SIZE=30              # Square map side length
NUM_AGENTS=2             # Number of vehicles
CAPACITY=30              # Vehicle capacity (fixed at 30 = DEMAND_NORM)
MAX_TIME=1000             # Max simulation time
MAX_END_TIME=1200        # Max deadline for static demands (when nodes disappear)

# --- Demand Generation (for Generator) ---
# TERMINOLOGY:
#   - NUM_NODES: Actual number of demand nodes (for tensor shapes)
#   - TOTAL_DEMAND: Upper limit of sum of all demands (capacity constraint, NOT node count)
NUM_NODES=20             # Number of demand nodes (reduce for limited VRAM)
TOTAL_DEMAND=60          # Upper limit of sum of all demands
MAX_C=5                  # Max demand per node (demands 1 to max_c)
MIN_LIFETIME=10          # Min demand lifetime
MAX_LIFETIME=50          # Max demand lifetime
RANDOMIZE_DEPOT="true"   # Randomize depot location ("true" or "false")

# --- Hardware ---
DEVICE="cuda"            # "cuda" or "cpu"
SEED=42                  # Random seed

# --- Checkpoints (optional, for loading pretrained models) ---
PLANNER_INITIALIZE="checkpoints/static_vrp_v2/best_n20.pt"    # Path to pretrained planner (leave empty if none)
# Path to pretrained diffusion generator checkpoint
# Recommended: use a supervised-trained model to avoid random initialization
# Example: "checkpoints/diffusion_model.pth"
GENERATOR_INITIALIZE="checkpoints/rl_generator/greedy_static_20251205-142015/best.pth"
RESUME_FROM=""           # Resume from checkpoint directory (leave empty if none)

# --- Output ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="checkpoints/cotrain/${MODE}_${TIMESTAMP}"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

echo "=============================================="
echo "Co-evolution Training Configuration"
echo "=============================================="
echo ""
echo "MODE:               ${MODE}"
echo "NUM_GPUS:           ${NUM_GPUS}"
echo ""
echo "CO-EVOLUTION:"
echo "  Cycles:           ${NUM_CYCLES}"
echo "  Planner epochs:   ${PLANNER_EPOCHS}"
if [[ -n "${FIRST_CYCLE_PLANNER_EPOCHS}" ]]; then
    echo "  First cycle epochs: ${FIRST_CYCLE_PLANNER_EPOCHS}"
fi
echo "  Generator epochs: ${GENERATOR_EPOCHS}"
echo ""
echo "BATCH SETTINGS:"
echo "  Batch size:       ${BATCH_SIZE}"
echo "  POMO size:        ${POMO_SIZE}"
echo "  Episodes/epoch:   ${EPISODES_PER_EPOCH}"
echo "  Version policy:   ${VERSION_POLICY}"
echo ""
echo "PROBLEM CACHE:"
echo "  Reuse ratio:      ${CACHE_REUSE_RATIO} (${CACHE_REUSE_RATIO}=80% from cache)"
echo "  Max per version:  ${MAX_PROBLEMS_PER_VERSION}"
echo "  Min for reuse:    ${MIN_CACHE_SIZE_FOR_REUSE}"
echo ""
echo "ENVIRONMENT:"
echo "  Map size:         ${MAP_SIZE}x${MAP_SIZE}"
echo "  Num agents:       ${NUM_AGENTS}"
echo "  Capacity:         ${CAPACITY}"
echo "  Max time:         ${MAX_TIME}"
echo "  Max end time:     ${MAX_END_TIME}"
echo ""
echo "DEMAND GENERATION:"
echo "  Num nodes:        ${NUM_NODES}"
echo "  Total demand:     ${TOTAL_DEMAND} (upper limit of sum of all demands)"
echo "  Max demand/node:  ${MAX_C}"
echo "  Lifetime:         ${MIN_LIFETIME}-${MAX_LIFETIME}"
echo "  Randomize depot:  ${RANDOMIZE_DEPOT}"
echo ""
echo "HARDWARE:"
echo "  Device:           ${DEVICE}"
echo "  Seed:             ${SEED}"
echo ""
echo "INITIALIZE:"
echo "  Save directory:   ${SAVE_DIR}"
if [[ -n "${PLANNER_INITIALIZE}" ]]; then
    echo "  Planner initialize:     ${PLANNER_INITIALIZE}"
fi
if [[ -n "${GENERATOR_INITIALIZE}" ]]; then
    echo "  Generator initialize:   ${GENERATOR_INITIALIZE}"
fi
if [[ -n "${RESUME_FROM}" ]]; then
    echo "  Resume from:      ${RESUME_FROM}"
fi
echo "=============================================="
echo ""

# ==============================================================================
# Build Command
# ==============================================================================

# Build common arguments
ARGS=(
    --mode "${MODE}"
    --num-cycles "${NUM_CYCLES}"
    --planner-epochs "${PLANNER_EPOCHS}"
    --generator-epochs "${GENERATOR_EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --pomo-size "${POMO_SIZE}"
    --episodes-per-epoch "${EPISODES_PER_EPOCH}"
    --version-policy "${VERSION_POLICY}"
    --latest-bias "${LATEST_BIAS}"
    --cache-reuse-ratio "${CACHE_REUSE_RATIO}"
    --max-problems-per-version "${MAX_PROBLEMS_PER_VERSION}"
    --min-cache-size-for-reuse "${MIN_CACHE_SIZE_FOR_REUSE}"
    --map-size "${MAP_SIZE}"
    --num-agents "${NUM_AGENTS}"
    --capacity "${CAPACITY}"
    --max-time "${MAX_TIME}"
    --max-end-time "${MAX_END_TIME}"
    --num-nodes "${NUM_NODES}"
    --total-demand "${TOTAL_DEMAND}"
    --max-c "${MAX_C}"
    --min-lifetime "${MIN_LIFETIME}"
    --max-lifetime "${MAX_LIFETIME}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --save-dir "${SAVE_DIR}"
    --num-gpus "${NUM_GPUS}"
)

# Add first cycle planner epochs if specified
if [[ -n "${FIRST_CYCLE_PLANNER_EPOCHS}" ]]; then
    ARGS+=(--first-cycle-planner-epochs "${FIRST_CYCLE_PLANNER_EPOCHS}")
fi

# Add planner early stopping if specified
if [[ -n "${PLANNER_EARLY_STOP_PATIENCE}" ]] && [[ "${PLANNER_EARLY_STOP_PATIENCE}" != "0" ]]; then
    ARGS+=(--planner-early-stop-patience "${PLANNER_EARLY_STOP_PATIENCE}")
fi
if [[ -n "${PLANNER_EARLY_STOP_THRESHOLD}" ]]; then
    ARGS+=(--planner-early-stop-threshold "${PLANNER_EARLY_STOP_THRESHOLD}")
fi

# Add generator early stopping if specified
if [[ -n "${GENERATOR_EARLY_STOP_PATIENCE}" ]] && [[ "${GENERATOR_EARLY_STOP_PATIENCE}" != "0" ]]; then
    ARGS+=(--generator-early-stop-patience "${GENERATOR_EARLY_STOP_PATIENCE}")
fi
if [[ -n "${GENERATOR_EARLY_STOP_THRESHOLD}" ]]; then
    ARGS+=(--generator-early-stop-threshold "${GENERATOR_EARLY_STOP_THRESHOLD}")
fi

# Add optional flags
if [[ "${RANDOMIZE_DEPOT}" == "true" ]]; then
    ARGS+=(--randomize-depot)
fi

# Add optional checkpoints (initialization paths)
if [[ -n "${PLANNER_INITIALIZE}" ]]; then
    ARGS+=(--planner-checkpoint "${PLANNER_INITIALIZE}")
fi

if [[ -n "${GENERATOR_INITIALIZE}" ]]; then
    ARGS+=(--generator-checkpoint "${GENERATOR_INITIALIZE}")
fi

if [[ -n "${RESUME_FROM}" ]]; then
    ARGS+=(--resume "${RESUME_FROM}")
fi

# =============================================================================
# Run Training
# =============================================================================

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    echo "Launching multi-GPU training with ${NUM_GPUS} GPUs using torchrun..."
    torchrun --nproc_per_node="${NUM_GPUS}" -m adversarial_v2.cotrain "${ARGS[@]}"
else
    echo "Launching single-GPU training..."
    python3 -m adversarial_v2.cotrain "${ARGS[@]}"
fi

echo ""
echo "=============================================="
echo "Training complete!"
echo "Checkpoints saved to: ${SAVE_DIR}"
echo "=============================================="
