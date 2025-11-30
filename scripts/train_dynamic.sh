#!/usr/bin/env bash
# Train Dynamic VRP Model (Static Model + Adapter)
#
# FIXED PARAMETERS (DO NOT CHANGE):
#   - Vehicle capacity: 30 (model sees capacity/30 = 1.0)
#   - Max demand per node: 5 (model sees demand/30 ∈ [0.033, 0.167])
#
# CONFIGURABLE PARAMETERS:
#   - NUM_DEMANDS: Number of customer nodes (20, 30, 50, etc.)
#   - GRID_SIZE: Map size for coordinate normalization (20, 30, 40, etc.)
#   - NUM_AGENTS: Number of vehicles (2, 3, 5, etc.)
#
# Usage Examples:
#   # Basic training with 20 demands
#   bash scripts/train_dynamic.sh
#
#   # Train with 50 demands on 40x40 map
#   NUM_DEMANDS=50 GRID_SIZE=40 bash scripts/train_dynamic.sh
#
#   # Train with different number of agents
#   NUM_AGENTS=3 NUM_DEMANDS=30 bash scripts/train_dynamic.sh

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# === Configurable Parameters ===
NUM_AGENTS="${NUM_AGENTS:-2}"             # Number of vehicles
NUM_DEMANDS="${NUM_DEMANDS:-20}"          # Number of customer nodes
GRID_SIZE="${GRID_SIZE:-20}"              # Map grid size (COORD_NORM)

# Dynamic VRP training specific
STATIC_CKPT="${STATIC_CKPT:-checkpoints/static_vrp_v2/best_n${NUM_DEMANDS}.pt}"
DYNAMIC_MODE="${DYNAMIC_MODE:-rl}"
EPOCHS="${EPOCHS:-30}"
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-50}"
SAVE_DIR="${SAVE_DIR:-checkpoints/dynamic_adapter_v2}"
DEVICE="${DEVICE:-cuda}"
BALANCE_WEIGHT="${BALANCE_WEIGHT:-0.5}"
USE_BALANCE_TRAINING="${USE_BALANCE_TRAINING:-false}"

echo "=== Dynamic VRP Training Configuration ==="
echo ""
echo "  FIXED (do not change):"
echo "    Vehicle capacity:   30"
echo "    Max demand/node:    5"
echo ""
echo "  CONFIGURABLE:"
echo "    Num agents:         $NUM_AGENTS"
echo "    Num demands:        $NUM_DEMANDS"
echo "    Grid size:          ${GRID_SIZE}x${GRID_SIZE}"
echo ""
echo "  TRAINING:"
echo "    Static checkpoint:  $STATIC_CKPT"
echo "    Mode:               $DYNAMIC_MODE"
echo "    Epochs:             $EPOCHS"
echo "    Episodes/epoch:     $EPISODES_PER_EPOCH"
echo "    Save directory:     $SAVE_DIR"
echo "    Device:             $DEVICE"
echo ""

# Validate checkpoint file exists before running
if [[ ! -f "$STATIC_CKPT" ]]; then
    echo "ERROR: Static checkpoint file not found: $STATIC_CKPT"
    exit 1
fi

cmd=(
    python -m training_v2.train_dynamic
    --static-checkpoint "$STATIC_CKPT"
    --mode "$DYNAMIC_MODE"
    --num-agents "$NUM_AGENTS"
    --num-demands "$NUM_DEMANDS"
    --grid-size "$GRID_SIZE"
    --epochs "$EPOCHS"
    --episodes-per-epoch "$EPISODES_PER_EPOCH"
    --save-dir "$SAVE_DIR"
    --device "$DEVICE"
    --balance-weight "$BALANCE_WEIGHT"
)

if [[ "$USE_BALANCE_TRAINING" == "true" ]]; then
    cmd+=(--use-balance-training)
fi

cmd+=("$@")

"${cmd[@]}"
