#!/usr/bin/env bash

# =============================================================================
# Train Diffusion V2 Generator
# Usage: bash scripts/train_diffusion_v2.sh
#
# This script trains the new VRP Diffusion Generator using PPO.
# Configuration is read from static_config.py
# =============================================================================

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# =============================================================================
# Read configuration from static_config.py
# =============================================================================
get_config() {
    python3 -c "from static_config import $1; print($1)" 2>/dev/null || echo ""
}

# --- Training Settings (read from static_config.py) ---
EPOCHS=$(get_config "DIFFUSION_V2_EPOCHS")

EPISODES_PER_EPOCH=$(get_config "DIFFUSION_V2_EPISODES_PER_EPOCH")

LR=$(get_config "DIFFUSION_V2_LR")

# --- VRP Problem Settings (read from static_config.py) ---
NUM_NODES=$(get_config "NUM_NODES")

MAP_SIZE=$(get_config "MAP_SIZE")

TOTAL_DEMAND=$(get_config "TOTAL_DEMAND")

MAX_C=$(get_config "MAX_C")

CAPACITY=$(get_config "CAPACITY")

# --- Depot Position (read from static_config.py) ---
DEPOT_X=$(get_config "DIFFUSION_V2_DEPOT_X")
DEPOT_Y=$(get_config "DIFFUSION_V2_DEPOT_Y")

# --- Randomize Depot (read from static_config.py) ---
RANDOMIZE_DEPOT=$(get_config "RANDOMIZE_DEPOT")

# --- Sampling Settings (read from static_config.py) ---
DDIM_STEPS=$(get_config "DIFFUSION_V2_DDIM_STEPS")

# --- Output Settings (read from static_config.py) ---
OUTPUT_DIR=$(get_config "DIFFUSION_V2_OUTPUT_DIR")

LOG_DIR=$(get_config "DIFFUSION_V2_LOG_DIR")

# --- Hardware (read from static_config.py) ---
DEVICE=$(get_config "DEVICE")

SEED=$(get_config "SEED")

# --- Resume (read from static_config.py) ---
RESUME=$(get_config "DIFFUSION_V2_RESUME")

# =============================================================================
# Display Configuration
# =============================================================================

echo "=========================================="
echo "Diffusion V2 Training Configuration"
echo "=========================================="
echo ""
echo "  TRAINING:"
echo "    Epochs:             $EPOCHS"
echo "    Episodes/epoch:     $EPISODES_PER_EPOCH"
echo "    Learning rate:      $LR"
echo ""
echo "  VRP PROBLEM:"
echo "    Num nodes:          $NUM_NODES"
echo "    Map size:           ${MAP_SIZE}x${MAP_SIZE}"
echo "    Total demand:       $TOTAL_DEMAND"
echo "    Max C:              $MAX_C"
echo "    Capacity:           $CAPACITY"
echo "    Depot:              ($DEPOT_X, $DEPOT_Y)"
echo "    Randomize Depot:    ${RANDOMIZE_DEPOT:-disabled}"
echo ""
echo "  SAMPLING:"
echo "    DDIM steps:         $DDIM_STEPS"
echo ""
echo "  OUTPUT:"
echo "    Checkpoint dir:     $OUTPUT_DIR"
echo "    Log dir:            $LOG_DIR"
echo ""
echo "  HARDWARE:"
echo "    Device:             $DEVICE"
echo "    Seed:               $SEED"
echo ""
if [[ -n "$RESUME" ]]; then
    echo "  RESUME FROM:          $RESUME"
    echo ""
fi
echo "=========================================="
echo ""

# =============================================================================
# Build Command
# =============================================================================

cmd=(
    python3 -m diffusion_v2.train
    --epochs "$EPOCHS"
    --episodes-per-epoch "$EPISODES_PER_EPOCH"
    --lr "$LR"
    --num-nodes "$NUM_NODES"
    --map-size "$MAP_SIZE"
    --total-demand "$TOTAL_DEMAND"
    --max-c "$MAX_C"
    --capacity "$CAPACITY"
    --depot-x "$DEPOT_X"
    --depot-y "$DEPOT_Y"
    --ddim-steps "$DDIM_STEPS"
    --output-dir "$OUTPUT_DIR"
    --log-dir "$LOG_DIR"
    --device "$DEVICE"
    --seed "$SEED"
)

if [[ -n "$RESUME" ]]; then
    cmd+=(--resume "$RESUME")
fi


if [[ "$RANDOMIZE_DEPOT" == "True" ]]; then
    RANDOMIZE_DEPOT_FLAG="--randomize-depot"
    cmd+=("$RANDOMIZE_DEPOT_FLAG")
else
    RANDOMIZE_DEPOT_FLAG=""
fi


echo "Starting training..."
"${cmd[@]}"
