#!/usr/bin/env bash
# Train Static VRP Model (POMO-style)
#
# FIXED PARAMETERS (DO NOT CHANGE):
#   - Vehicle capacity: 30 (model sees capacity/30 = 1.0)
#   - Max demand per node: 5 (model sees demand/30 ∈ [0.033, 0.167])
#
# CONFIGURABLE PARAMETERS:
#   - NUM_NODES: Number of demand nodes
#   - TOTAL_DEMAND: Upper limit of sum of all customer demands (capacity constraint)
#   - MAP_SIZE: Side length of the square map (map is MAP_SIZE × MAP_SIZE)
#   - NUM_AGENTS: Number of vehicles (2, 3, 5, etc.)
#
# Usage Examples:
#   # Basic training with default settings
#   bash scripts/train_static.sh
#
#   # Train with more nodes on larger map
#   NUM_NODES=50 MAP_SIZE=40 bash scripts/train_static.sh
#
#   # Train with more epochs
#   EPOCHS=1000 NUM_NODES=30 bash scripts/train_static.sh
#
#   # Resume from checkpoint
#   RESUME_FROM=checkpoints/static_vrp_v2/checkpoint_n20_ep100.pt bash scripts/train_static.sh

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# === Configurable Parameters ===
NUM_AGENTS="${NUM_AGENTS:-10}"             # Number of vehicles

# NUM_NODES: Number of demand nodes
NUM_NODES="${NUM_NODES:-20}"
# TOTAL_DEMAND: Upper limit of sum of all customer demands (capacity constraint)
TOTAL_DEMAND="${TOTAL_DEMAND:-80}"
MAP_SIZE="${MAP_SIZE:-50}"                 # Square map side length (map is MAP_SIZE × MAP_SIZE)

# Static VRP training specific
TARGET_VEHICLES="${TARGET_VEHICLES:-$NUM_AGENTS}"  # Use NUM_AGENTS by default

# Training params
POMO_SIZE="${POMO_SIZE:-100}"             # Parallel rollouts (50-100 recommended)
AUG_FACTOR="${AUG_FACTOR:-1}"             # Data augmentation (1 or 8)
EPOCHS="${EPOCHS:-500}"                   # Training epochs (500-2000 for good results)
EPISODES_PER_EPOCH="${EPISODES_PER_EPOCH:-10000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
SAVE_DIR="${SAVE_DIR:-checkpoints/static_vrp_v2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
DEVICE="${DEVICE:-cuda}"

# Resume training (optional)
RESUME_FROM="${RESUME_FROM:-}"

# Model architecture
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
ENCODER_LAYERS="${ENCODER_LAYERS:-6}"
HEADS="${HEADS:-8}"

echo "=== Static VRP Training Configuration ==="
echo ""
echo "  FIXED (do not change):"
echo "    Vehicle capacity:   30"
echo "    Max demand/node:    5"
echo ""
echo "  CONFIGURABLE:"
echo "    Num nodes:          $NUM_NODES"
echo "    Total demand:       $TOTAL_DEMAND (upper limit of sum of all demands)"
echo "    Map size:           ${MAP_SIZE}x${MAP_SIZE}"
echo "    Target vehicles:    $TARGET_VEHICLES"
echo ""
echo "  TRAINING:"
echo "    POMO size:          $POMO_SIZE (parallel rollouts)"
echo "    Epochs:             $EPOCHS"
echo "    Episodes/epoch:     $EPISODES_PER_EPOCH"
echo "    Batch size:         $BATCH_SIZE"
echo "    Learning rate:      $LR"
echo "    Save directory:     $SAVE_DIR"
echo "    Device:             $DEVICE"
if [[ -n "$RESUME_FROM" ]]; then
    echo "    Resume from:        $RESUME_FROM"
fi
echo ""
echo "  Total training episodes: $((EPOCHS * EPISODES_PER_EPOCH))"
echo ""

# Validate resume checkpoint exists if specified
if [[ -n "$RESUME_FROM" && ! -f "$RESUME_FROM" ]]; then
    echo "ERROR: Resume checkpoint file not found: $RESUME_FROM"
    exit 1
fi

cmd=(
    python3 -m training_v2.train_static
    --num-nodes "$NUM_NODES"
    --target-vehicles "$TARGET_VEHICLES"
    --pomo-size "$POMO_SIZE"
    --embedding-dim "$EMBEDDING_DIM"
    --encoder-layers "$ENCODER_LAYERS"
    --heads "$HEADS"
    --epochs "$EPOCHS"
    --episodes-per-epoch "$EPISODES_PER_EPOCH"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --save-dir "$SAVE_DIR"
    --device "$DEVICE"
)

if [[ -n "$RESUME_FROM" ]]; then
    cmd+=(--resume "$RESUME_FROM")
fi

# Pass any additional arguments
cmd+=("$@")

"${cmd[@]}"
