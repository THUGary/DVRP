#!/usr/bin/env bash
# =============================================================================
# Train Static VRP Model (POMO-style)
# Usage: bash scripts/train_static.sh
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
NUM_AGENTS=2              # Number of vehicles

# --- Demand Settings ---
NUM_NODES=20               # Number of demand nodes
TOTAL_DEMAND=60            # Upper limit of sum of all customer demands (capacity constraint)

# --- Environment Settings ---
MAP_SIZE=30                # Square map side length (map is MAP_SIZE × MAP_SIZE)
TARGET_VEHICLES=2         # Target number of vehicles (usually same as NUM_AGENTS)

# --- Training Parameters ---
EPOCHS=2000                 # Training epochs (500-2000 for good results)
EPISODES_PER_EPOCH=10000   # Episodes per epoch
BATCH_SIZE=64              # Batch size
LR="1e-4"                  # Learning rate
POMO_SIZE=100              # Parallel rollouts (50-100 recommended)
AUG_FACTOR=1               # Data augmentation (1 or 8)

PATIENCE=50                # Early stopping patience
THRESHOLD=0.0001            # Early stopping improvement threshold

# --- Model Architecture ---
EMBEDDING_DIM=128
ENCODER_LAYERS=6
HEADS=8

# --- Hardware ---
DEVICE="cuda"              # "cuda" or "cpu"

# --- Output Settings ---
SAVE_DIR="checkpoints/static_vrp_v2"
SAVE_INTERVAL=10           # Save checkpoint every N epochs

# --- Resume Training (leave empty to train from scratch) ---
RESUME_FROM=""

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

echo "=========================================="
echo "Static VRP Training Configuration"
echo "=========================================="
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
echo "    Early stopping patience: $PATIENCE"
echo "    Early stopping threshold: $THRESHOLD"
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
    --patience "$PATIENCE"
    --threshold "$THRESHOLD"
)

if [[ -n "$RESUME_FROM" ]]; then
    cmd+=(--resume "$RESUME_FROM")
fi

"${cmd[@]}"
