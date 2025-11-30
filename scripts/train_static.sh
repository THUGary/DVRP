#!/usr/bin/env bash
# Train Static VRP Model (POMO-style)
#
# FIXED PARAMETERS (DO NOT CHANGE):
#   - Vehicle capacity: 30 (model sees capacity/30 = 1.0)
#   - Max demand per node: 5 (model sees demand/30 ∈ [0.033, 0.167])
#
# CONFIGURABLE PARAMETERS:
#   - PROBLEM_SIZE: Number of customer nodes (20, 30, 50, etc.)
#   - MAP_SIZE: Grid size for coordinate normalization (20, 30, 40, etc.)
#   - NUM_AGENTS: Number of vehicles (2, 3, 5, etc.)
#
# Usage Examples:
#   # Basic training with 20 nodes
#   bash scripts/train_static.sh
#
#   # Train with 50 nodes on 40x40 map
#   PROBLEM_SIZE=50 MAP_SIZE=40 bash scripts/train_static.sh
#
#   # Train with more epochs and different problem size
#   EPOCHS=1000 PROBLEM_SIZE=30 bash scripts/train_static.sh
#
#   # Resume from checkpoint
#   RESUME_FROM=checkpoints/static_vrp_v2/checkpoint_n20_ep100.pt bash scripts/train_static.sh

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# === Configurable Parameters ===
NUM_AGENTS="${NUM_AGENTS:-10}"             # Number of vehicles
# PROBLEM_SIZE = TOTAL_DEMAND = Number of customer nodes to visit
# Both names are supported for compatibility
PROBLEM_SIZE="${PROBLEM_SIZE:-${TOTAL_DEMAND:-80}}"  # Number of customer nodes
MAP_SIZE="${MAP_SIZE:-50}"                # Grid size (COORD_NORM)

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
echo "    Problem size:       $PROBLEM_SIZE nodes"
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
    --problem-size "$PROBLEM_SIZE"
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
