#!/usr/bin/env bash
# Train Dynamic VRP Model (Static Model + Adapter)
#
# FIXED PARAMETERS (DO NOT CHANGE):
#   - Vehicle capacity: 30 (model sees capacity/30 = 1.0)
#   - Max demand per node: 5 (model sees demand/30 ∈ [0.033, 0.167])
#
# CONFIGURABLE PARAMETERS:
#   - NUM_NODES: Number of demand nodes (exact count for tensor shapes)
#                This is the actual number of customer nodes to generate
#   - MAP_SIZE: Side length of the square map (map is MAP_SIZE × MAP_SIZE)
#   - NUM_AGENTS: Number of vehicles (2, 3, 5, etc.)
#
# TERMINOLOGY:
#   - num_nodes: Actual number of demand nodes (for tensor shapes)
#   - total_demand: Upper limit of sum of all demands (NOT node count, NOT used here)
#
# Usage Examples:
#   # Basic training with default settings
#   bash scripts/train_dynamic.sh
#
#   # Train with more nodes on larger map
#   NUM_NODES=50 MAP_SIZE=40 bash scripts/train_dynamic.sh
#
#   # Train with different number of agents
#   NUM_AGENTS=3 NUM_NODES=30 bash scripts/train_dynamic.sh

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# === Configurable Parameters ===
NUM_AGENTS="${NUM_AGENTS:-2}"             # Number of vehicles
# NUM_NODES: Actual number of demand nodes (NOT total_demand)
NUM_NODES="${NUM_NODES:-20}"              # Number of demand nodes
MAP_SIZE="${MAP_SIZE:-20}"                # Square map side length (map is MAP_SIZE × MAP_SIZE)

# Dynamic VRP training specific
STATIC_CKPT="${STATIC_CKPT:-checkpoints/static_vrp_v2/best_n${NUM_NODES}.pt}"
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
echo "    Num nodes:          $NUM_NODES (actual demand node count)"
echo "    Map size:           ${MAP_SIZE}x${MAP_SIZE}"
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
    python3 -m training_v2.train_dynamic
    --static-checkpoint "$STATIC_CKPT"
    --mode "$DYNAMIC_MODE"
    --num-agents "$NUM_AGENTS"
    --num-nodes "$NUM_NODES"
    --map-size "$MAP_SIZE"
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
