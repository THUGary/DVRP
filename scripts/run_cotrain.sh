#!/usr/bin/env bash

# =============================================================================
# Run Co-Training (Alternating Generator and Planner Training)
# Usage: bash scripts/run_cotrain.sh
#
# Configuration is read from static_config.py
# =============================================================================

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# =============================================================================
# Read configuration from static_config.py
# =============================================================================
get_config() {
    python3 -c "from static_config import $1; print($1)"
}

# --- Co-Training Settings ---
NUM_CYCLES=$(get_config "NUM_CYCLES")
MODE=$(get_config "MODE")
NUM_GPUS=$(get_config "NUM_GPUS")

# --- Generator Training ---
GENERATOR_EPOCHS=$(get_config "GENERATOR_EPOCHS")

# --- Planner Training ---
PLANNER_EPOCHS=$(get_config "PLANNER_EPOCHS")
FIRST_CYCLE_PLANNER_EPOCHS=$(get_config "FIRST_CYCLE_PLANNER_EPOCHS")

# --- Batch and Episode Settings ---
BATCH_SIZE=$(get_config "BATCH_SIZE")
EPISODES_PER_EPOCH=$(get_config "EPISODES_PER_EPOCH")
POMO_SIZE=$(get_config "POMO_SIZE")

# --- Environment Settings ---
NUM_AGENTS=$(get_config "NUM_AGENTS")
NUM_NODES=$(get_config "NUM_NODES")
TOTAL_DEMAND=$(get_config "TOTAL_DEMAND")
MAP_SIZE=$(get_config "MAP_SIZE")
MAX_C=$(get_config "MAX_C")
CAPACITY=$(get_config "CAPACITY")
MAX_TIME=$(get_config "MAX_TIME")
MIN_LIFETIME=$(get_config "MIN_LIFETIME")
MAX_LIFETIME=$(get_config "MAX_LIFETIME")
RANDOMIZE_DEPOT=$(get_config "RANDOMIZE_DEPOT")

# --- Early Stopping ---
PLANNER_EARLY_STOP_PATIENCE=$(get_config "PLANNER_EARLY_STOP_PATIENCE")
PLANNER_EARLY_STOP_THRESHOLD=$(get_config "PLANNER_EARLY_STOP_THRESHOLD")
GENERATOR_EARLY_STOP_PATIENCE=$(get_config "GENERATOR_EARLY_STOP_PATIENCE")
GENERATOR_EARLY_STOP_THRESHOLD=$(get_config "GENERATOR_EARLY_STOP_THRESHOLD")

# --- Problem Cache Settings ---
CACHE_REUSE_RATIO=$(get_config "CACHE_REUSE_RATIO")
MAX_PROBLEMS_PER_VERSION=$(get_config "MAX_PROBLEMS_PER_VERSION")
MIN_CACHE_SIZE_FOR_REUSE=$(get_config "MIN_CACHE_SIZE_FOR_REUSE")

# --- Version Sampling ---
VERSION_POLICY=$(get_config "VERSION_POLICY")
LATEST_BIAS=$(get_config "LATEST_BIAS")

# --- Hardware ---
DEVICE=$(get_config "DEVICE")
SEED=$(get_config "SEED")

# --- Checkpoints ---
PLANNER_INITIALIZE=$(get_config "PLANNER_INITIALIZE")
GENERATOR_INITIALIZE=$(get_config "GENERATOR_INITIALIZE")
COTRAIN_RESUME_FROM=$(get_config "COTRAIN_RESUME_FROM")

# =============================================================================
# Display Configuration
# =============================================================================

echo "=========================================="
echo "Co-Training Configuration"
echo "=========================================="
echo ""
echo "  CO-TRAINING:"
echo "    Mode:               $MODE"
echo "    Cycles:             $NUM_CYCLES"
echo "    Num GPUs:           $NUM_GPUS"
if [[ -n "$COTRAIN_RESUME_FROM" ]]; then
    echo "    Resume from:        $COTRAIN_RESUME_FROM"
fi
echo ""
echo "  TRAINING:"
echo "    Generator epochs:   $GENERATOR_EPOCHS"
echo "    Planner epochs:     $PLANNER_EPOCHS"
echo "    First cycle epochs: $FIRST_CYCLE_PLANNER_EPOCHS"
echo "    Batch size:         $BATCH_SIZE"
echo "    Episodes/epoch:     $EPISODES_PER_EPOCH"
echo "    POMO size:          $POMO_SIZE"
echo ""
echo "  ENVIRONMENT:"
echo "    Num agents:         $NUM_AGENTS"
echo "    Num nodes:          $NUM_NODES"
echo "    Total demand:       $TOTAL_DEMAND"
echo "    Map size:           ${MAP_SIZE}x${MAP_SIZE}"
echo "    Max C:              $MAX_C"
echo "    Capacity:           $CAPACITY"
echo "    Max time:           $MAX_TIME"
echo ""
echo "  HARDWARE:"
echo "    Device:             $DEVICE"
echo "    Seed:               $SEED"
echo ""

# =============================================================================
# Validate Checkpoints
# =============================================================================

if [[ -n "$GENERATOR_INITIALIZE" && ! -f "$GENERATOR_INITIALIZE" ]]; then
    echo "ERROR: Generator checkpoint not found: $GENERATOR_INITIALIZE"
    exit 1
fi

if [[ -n "$PLANNER_INITIALIZE" && ! -f "$PLANNER_INITIALIZE" ]]; then
    echo "ERROR: Planner checkpoint not found: $PLANNER_INITIALIZE"
    exit 1
fi

if [[ -n "$COTRAIN_RESUME_FROM" && ! -d "$COTRAIN_RESUME_FROM" ]]; then
    echo "ERROR: Resume directory not found: $COTRAIN_RESUME_FROM"
    exit 1
fi

# =============================================================================
# Build Command
# =============================================================================

cmd=(
    python3 -m adversarial_v2.cotrain
    --mode "$MODE"
    --num-cycles "$NUM_CYCLES"
    --num-gpus "$NUM_GPUS"
    --num-agents "$NUM_AGENTS"
    --num-nodes "$NUM_NODES"
    --total-demand "$TOTAL_DEMAND"
    --map-size "$MAP_SIZE"
    --max-c "$MAX_C"
    --capacity "$CAPACITY"
    --max-time "$MAX_TIME"
    --min-lifetime "$MIN_LIFETIME"
    --max-lifetime "$MAX_LIFETIME"
    --pomo-size "$POMO_SIZE"
    --device "$DEVICE"
    --seed "$SEED"
    --generator-epochs "$GENERATOR_EPOCHS"
    --planner-epochs "$PLANNER_EPOCHS"
    --batch-size "$BATCH_SIZE"
    --episodes-per-epoch "$EPISODES_PER_EPOCH"
    --planner-early-stop-patience "$PLANNER_EARLY_STOP_PATIENCE"
    --planner-early-stop-threshold "$PLANNER_EARLY_STOP_THRESHOLD"
    --generator-early-stop-patience "$GENERATOR_EARLY_STOP_PATIENCE"
    --generator-early-stop-threshold "$GENERATOR_EARLY_STOP_THRESHOLD"
    --cache-reuse-ratio "$CACHE_REUSE_RATIO"
    --max-problems-per-version "$MAX_PROBLEMS_PER_VERSION"
    --min-cache-size-for-reuse "$MIN_CACHE_SIZE_FOR_REUSE"
    --version-policy "$VERSION_POLICY"
    --latest-bias "$LATEST_BIAS"
)

# Optional arguments
if [[ -n "$FIRST_CYCLE_PLANNER_EPOCHS" && "$FIRST_CYCLE_PLANNER_EPOCHS" != "None" ]]; then
    cmd+=(--first-cycle-planner-epochs "$FIRST_CYCLE_PLANNER_EPOCHS")
fi

if [[ "$RANDOMIZE_DEPOT" == "True" ]]; then
    cmd+=(--randomize-depot)
fi

if [[ -n "$GENERATOR_INITIALIZE" ]]; then
    cmd+=(--generator-checkpoint "$GENERATOR_INITIALIZE")
fi

if [[ -n "$PLANNER_INITIALIZE" ]]; then
    cmd+=(--planner-checkpoint "$PLANNER_INITIALIZE")
fi

if [[ -n "$COTRAIN_RESUME_FROM" ]]; then
    cmd+=(--resume "$COTRAIN_RESUME_FROM")
fi

"${cmd[@]}"
