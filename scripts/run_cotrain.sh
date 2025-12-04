#!/bin/bash
# =============================================================================
# Co-evolution training script for V2Planner and Diffusion Generator
# Usage: bash scripts/run_cotrain.sh
#
# Edit the configuration variables below to change settings.
# =============================================================================

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dvrp

# Go to project root
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

# =============================================================================
# CONFIGURATION - Edit these variables to change settings
# =============================================================================

# --- Training Mode ---
MODE="static"                    # "static" or "dynamic"

# --- Co-evolution Settings ---
NUM_CYCLES=5             # Number of co-evolution cycles
PLANNER_EPOCHS=20        # Planner training epochs per cycle
FIRST_CYCLE_PLANNER_EPOCHS=200  # First cycle planner epochs (leave empty to use PLANNER_EPOCHS)
GENERATOR_EPOCHS=4      # Generator training epochs per cycle

# --- Planner Early Stopping (within each cycle) ---
# Stop planner training early if score doesn't improve for PATIENCE epochs
# Set to 0 or empty to disable early stopping
PLANNER_EARLY_STOP_PATIENCE=15    # Number of epochs without improvement to trigger early stop
PLANNER_EARLY_STOP_THRESHOLD=0.01  # Minimum improvement threshold

# --- Batch Settings ---
BATCH_SIZE=32            # Batch size for training
POMO_SIZE=100            # POMO parallel rollouts
EPISODES_PER_EPOCH=128   # Episodes per epoch

# --- Version Sampling (to prevent policy cycling) ---
VERSION_POLICY="latest_biased"  # "uniform", "latest_biased", "all"
LATEST_BIAS=0.5          # P(sample latest) when latest_biased

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
NUM_NODES=30             # Number of demand nodes (reduce for limited VRAM)
TOTAL_DEMAND=100          # Upper limit of sum of all demands
MAX_C=5                  # Max demand per node (demands 1 to max_c)
MIN_LIFETIME=10          # Min demand lifetime
MAX_LIFETIME=50          # Max demand lifetime
RANDOMIZE_DEPOT="true"   # Randomize depot location ("true" or "false")

# --- Hardware ---
DEVICE="cuda"            # "cuda" or "cpu"
SEED=42                  # Random seed

# --- Checkpoints (optional, for loading pretrained models) ---
PLANNER_INITIALIZE="checkpoints/cotrain/static_20251203_104147/planner_cycle_1.pt"    # Path to pretrained planner (leave empty if none)
# Path to pretrained diffusion generator checkpoint
# Recommended: use a supervised-trained model to avoid random initialization
# Example: "checkpoints/diffusion_model.pth"
GENERATOR_INITIALIZE="checkpoints/diffusion_model.pth"
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

CMD=(
    python3 -m adversarial_v2.cotrain
    --mode "${MODE}"
    --num-cycles "${NUM_CYCLES}"
    --planner-epochs "${PLANNER_EPOCHS}"
    --generator-epochs "${GENERATOR_EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --pomo-size "${POMO_SIZE}"
    --episodes-per-epoch "${EPISODES_PER_EPOCH}"
    --version-policy "${VERSION_POLICY}"
    --latest-bias "${LATEST_BIAS}"
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
)

# Add first cycle planner epochs if specified
if [[ -n "${FIRST_CYCLE_PLANNER_EPOCHS}" ]]; then
    CMD+=(--first-cycle-planner-epochs "${FIRST_CYCLE_PLANNER_EPOCHS}")
fi

# Add planner early stopping if specified
if [[ -n "${PLANNER_EARLY_STOP_PATIENCE}" ]] && [[ "${PLANNER_EARLY_STOP_PATIENCE}" != "0" ]]; then
    CMD+=(--planner-early-stop-patience "${PLANNER_EARLY_STOP_PATIENCE}")
fi
if [[ -n "${PLANNER_EARLY_STOP_THRESHOLD}" ]]; then
    CMD+=(--planner-early-stop-threshold "${PLANNER_EARLY_STOP_THRESHOLD}")
fi

# Add optional flags
if [[ "${RANDOMIZE_DEPOT}" == "true" ]]; then
    CMD+=(--randomize-depot)
fi

# Add optional checkpoints (initialization paths)
if [[ -n "${PLANNER_INITIALIZE}" ]]; then
    CMD+=(--planner-checkpoint "${PLANNER_INITIALIZE}")
fi

if [[ -n "${GENERATOR_INITIALIZE}" ]]; then
    CMD+=(--generator-checkpoint "${GENERATOR_INITIALIZE}")
fi

if [[ -n "${RESUME_FROM}" ]]; then
    CMD+=(--resume "${RESUME_FROM}")
fi

# =============================================================================
# Run Training
# =============================================================================

"${CMD[@]}"

echo ""
echo "=============================================="
echo "Training complete!"
echo "Checkpoints saved to: ${SAVE_DIR}"
echo "=============================================="
