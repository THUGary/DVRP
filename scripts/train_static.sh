#!/usr/bin/env bash

# =============================================================================
# Train Static VRP Model (POMO-style)
# Usage: bash scripts/train_static.sh
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

# --- Environment Settings ---
NUM_NODES=$(get_config "NUM_NODES")
TARGET_VEHICLES=$(get_config "TARGET_VEHICLES")

# --- Training Parameters ---
EPOCHS=$(get_config "EPOCHS")
EPISODES_PER_EPOCH=$(get_config "TRAIN_EPISODES_PER_EPOCH")
BATCH_SIZE=$(get_config "TRAIN_BATCH_SIZE")
LR=$(get_config "LR")
POMO_SIZE=$(get_config "POMO_SIZE")
AUG_FACTOR=$(get_config "AUG_FACTOR")

# --- Early Stopping ---
PATIENCE=$(get_config "PATIENCE")
THRESHOLD=$(get_config "THRESHOLD")

# --- Model Architecture ---
EMBEDDING_DIM=$(get_config "EMBEDDING_DIM")
ENCODER_LAYERS=$(get_config "ENCODER_LAYERS")
HEADS=$(get_config "HEADS")

# --- Hardware ---
DEVICE=$(get_config "DEVICE")

# --- Output Settings ---
SAVE_DIR=$(get_config "SAVE_DIR")

# --- Resume Training ---
RESUME_FROM=$(get_config "RESUME_FROM")

# --- Pre-generated Dataset (optional) ---
TRAIN_DATA_PATH=$(get_config "TRAIN_DATA_PATH")

# =============================================================================
# Display Configuration
# =============================================================================

echo "=========================================="
echo "Static VRP Training Configuration"
echo "=========================================="
echo ""
echo "  ENVIRONMENT:"
echo "    Num nodes:          $NUM_NODES"
echo "    Target vehicles:    $TARGET_VEHICLES"
echo ""
echo "  TRAINING:"
echo "    POMO size:          $POMO_SIZE"
echo "    Aug factor:         $AUG_FACTOR"
echo "    Epochs:             $EPOCHS"
echo "    Episodes/epoch:     $EPISODES_PER_EPOCH"
echo "    Batch size:         $BATCH_SIZE"
echo "    Learning rate:      $LR"
echo ""
echo "  EARLY STOPPING:"
echo "    Patience:           $PATIENCE"
echo "    Threshold:          $THRESHOLD"
echo ""
echo "  MODEL:"
echo "    Embedding dim:      $EMBEDDING_DIM"
echo "    Encoder layers:     $ENCODER_LAYERS"
echo "    Heads:              $HEADS"
echo ""
echo "  OUTPUT:"
echo "    Save directory:     $SAVE_DIR"
echo "    Device:             $DEVICE"
if [[ -n "$RESUME_FROM" ]]; then
    echo "    Resume from:        $RESUME_FROM"
fi
if [[ -n "$TRAIN_DATA_PATH" ]]; then
    echo "    Train data:         $TRAIN_DATA_PATH"
else
    echo "    Train data:         (random generation)"
fi
echo ""

# Validate resume checkpoint
if [[ -n "$RESUME_FROM" && ! -f "$RESUME_FROM" ]]; then
    echo "ERROR: Resume checkpoint not found: $RESUME_FROM"
    exit 1
fi

# Validate train data file if specified
if [[ -n "$TRAIN_DATA_PATH" && ! -f "$TRAIN_DATA_PATH" ]]; then
    echo "WARNING: Train data file not found: $TRAIN_DATA_PATH"
    echo "         Will use random generation instead."
fi

# =============================================================================
# Build Command
# =============================================================================

cmd=(
    python3 -m training_v2.train_static
    --num-nodes "$NUM_NODES"
    --target-vehicles "$TARGET_VEHICLES"
    --pomo-size "$POMO_SIZE"
    --aug-factor "$AUG_FACTOR"
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

if [[ -n "$TRAIN_DATA_PATH" ]]; then
    cmd+=(--train-data "$TRAIN_DATA_PATH")
fi

"${cmd[@]}"
