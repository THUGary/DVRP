#!/bin/bash
# Co-evolution training script for V2Planner and Diffusion Generator
#
# All configuration is defined here and passed via command line arguments.
# This ensures a single source of truth for all parameters.
#
# Usage:
#   bash scripts/run_cotrain.sh                    # Use defaults below
#   MODE=dynamic bash scripts/run_cotrain.sh      # Override mode
#   NUM_CYCLES=20 TOTAL_DEMAND=150 bash scripts/run_cotrain.sh  # Override multiple
#
# Parameters can be overridden via environment variables (see below)

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate drl

# Go to project root
cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

# ==============================================================================
# ALL CONFIGURATION PARAMETERS (override via environment variables)
# ==============================================================================

# --- Training Mode ---
MODE="${MODE:-static}"                    # "static" or "dynamic"

# --- Co-evolution Settings ---
NUM_CYCLES="${NUM_CYCLES:-5}"             # Number of co-evolution cycles
PLANNER_EPOCHS="${PLANNER_EPOCHS:-20}"    # Planner training epochs per cycle
GENERATOR_EPOCHS="${GENERATOR_EPOCHS:-10}" # Generator training epochs per cycle

# --- Batch Settings ---
BATCH_SIZE="${BATCH_SIZE:-64}"            # Batch size for training
POMO_SIZE="${POMO_SIZE:-100}"             # POMO parallel rollouts
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-100}"  # Episodes per epoch

# --- Version Sampling (to prevent policy cycling) ---
VERSION_POLICY="${VERSION_POLICY:-latest_biased}"  # "uniform", "latest_biased", "all"
LATEST_BIAS="${LATEST_BIAS:-0.7}"         # P(sample latest) when latest_biased

# --- Environment Settings ---
# map_size: Side length of the square map (map is map_size × map_size)
MAP_SIZE="${MAP_SIZE:-20}"                # Square map side length
NUM_AGENTS="${NUM_AGENTS:-2}"             # Number of vehicles
CAPACITY="${CAPACITY:-30}"                # Vehicle capacity (fixed at 30 = DEMAND_NORM)
MAX_TIME="${MAX_TIME:-100}"               # Max simulation time

# --- Demand Generation (for Generator) ---
# TERMINOLOGY:
#   - NUM_NODES: Actual number of demand nodes (for tensor shapes)
#   - TOTAL_DEMAND: Upper limit of sum of all demands (capacity constraint, NOT node count)
NUM_NODES="${NUM_NODES:-20}"              # Number of demand nodes (reduce for limited VRAM)
TOTAL_DEMAND="${TOTAL_DEMAND:-60}"        # Upper limit of sum of all demands
MAX_C="${MAX_C:-5}"                       # Max demand per node (demands 1 to max_c)
MIN_LIFETIME="${MIN_LIFETIME:-10}"        # Min demand lifetime
MAX_LIFETIME="${MAX_LIFETIME:-50}"        # Max demand lifetime
RANDOMIZE_DEPOT="${RANDOMIZE_DEPOT:-true}" # Randomize depot location

# --- Hardware ---
DEVICE="${DEVICE:-cuda}"                  # "cuda" or "cpu"
SEED="${SEED:-42}"                        # Random seed

# --- Checkpoints (optional, for loading pretrained models) ---
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-}"      # Path to pretrained planner
GENERATOR_CHECKPOINT="${GENERATOR_CHECKPOINT:-}"  # Path to pretrained generator
RESUME_FROM="${RESUME_FROM:-}"            # Resume from checkpoint directory

# --- Output ---
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="${SAVE_DIR:-checkpoints/cotrain/${MODE}_${TIMESTAMP}}"

# ==============================================================================
# Print Configuration
# ==============================================================================

echo "=============================================="
echo "Co-evolution Training Configuration"
echo "=============================================="
echo ""
echo "MODE:               ${MODE}"
echo ""
echo "CO-EVOLUTION:"
echo "  Cycles:           ${NUM_CYCLES}"
echo "  Planner epochs:   ${PLANNER_EPOCHS}"
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
echo "OUTPUT:"
echo "  Save directory:   ${SAVE_DIR}"
if [[ -n "${PLANNER_CHECKPOINT}" ]]; then
    echo "  Planner ckpt:     ${PLANNER_CHECKPOINT}"
fi
if [[ -n "${GENERATOR_CHECKPOINT}" ]]; then
    echo "  Generator ckpt:   ${GENERATOR_CHECKPOINT}"
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
    --num-nodes "${NUM_NODES}"
    --total-demand "${TOTAL_DEMAND}"
    --max-c "${MAX_C}"
    --min-lifetime "${MIN_LIFETIME}"
    --max-lifetime "${MAX_LIFETIME}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --save-dir "${SAVE_DIR}"
)

# Add optional flags
if [[ "${RANDOMIZE_DEPOT}" == "true" ]]; then
    CMD+=(--randomize-depot)
fi

# Add optional checkpoints
if [[ -n "${PLANNER_CHECKPOINT}" ]]; then
    CMD+=(--planner-checkpoint "${PLANNER_CHECKPOINT}")
fi

if [[ -n "${GENERATOR_CHECKPOINT}" ]]; then
    CMD+=(--generator-checkpoint "${GENERATOR_CHECKPOINT}")
fi

if [[ -n "${RESUME_FROM}" ]]; then
    CMD+=(--resume "${RESUME_FROM}")
fi

# Add any extra arguments passed to this script
CMD+=("$@")

# ==============================================================================
# Run Training
# ==============================================================================

"${CMD[@]}"

echo ""
echo "=============================================="
echo "Training complete!"
echo "Checkpoints saved to: ${SAVE_DIR}"
echo "=============================================="
