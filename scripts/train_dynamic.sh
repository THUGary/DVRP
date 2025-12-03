#!/usr/bin/env bash
# =============================================================================
# Train Dynamic VRP Model (Static Model + Adapter)
# Usage: bash scripts/train_dynamic.sh
#
# Edit the configuration variables below to change settings.
#
# FIXED PARAMETERS (DO NOT CHANGE - model training assumptions):
#   - Vehicle capacity: 30 (model sees capacity/30 = 1.0)
#   - Max demand per node: 5 (model sees demand/30 ∈ [0.033, 0.167])
# =============================================================================

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# =============================================================================
# CONFIGURATION - Edit these variables to change settings
# =============================================================================

# --- Agent Settings ---
NUM_AGENTS=2               # Number of vehicles

# --- Demand Settings ---
NUM_NODES=20               # Number of demand nodes

# --- Environment Settings ---
MAP_SIZE=20                # Square map side length (map is MAP_SIZE × MAP_SIZE)

# --- Training Parameters ---
DYNAMIC_MODE="rl"          # Training mode
EPOCHS=30                  # Training epochs
EPISODES_PER_EPOCH=50      # Episodes per epoch
BALANCE_WEIGHT=0.5         # Balance weight for multi-agent
USE_BALANCE_TRAINING="false"  # Use balance training ("true" or "false")

# --- Hardware ---
DEVICE="cuda"              # "cuda" or "cpu"

# --- Checkpoints ---
# Static checkpoint (should match NUM_NODES)
STATIC_CKPT="checkpoints/static_vrp_v2/best_n${NUM_NODES}.pt"
SAVE_DIR="checkpoints/dynamic_adapter_v2"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

echo "=========================================="
echo "Dynamic VRP Training Configuration"
echo "=========================================="
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
echo "=========================================="

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

"${cmd[@]}"
