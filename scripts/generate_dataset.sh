#!/usr/bin/env bash

# =============================================================================
# Generate Static VRP Dataset
# Usage: bash scripts/generate_dataset.sh
#
# This script generates pre-computed training and test datasets for static VRP
# training. It supports two modes:
#   - random: Generate problems with uniform random distribution
#   - diffusion: Generate problems using a trained diffusion model
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

# --- Dataset Generation Settings ---
OUTPUT_DIR=$(get_config "DATASET_OUTPUT_DIR")
TOTAL_EPISODES=$(get_config "DATASET_TOTAL_EPISODES")
TEST_RATIO=$(get_config "DATASET_TEST_RATIO")
MODE=$(get_config "DATASET_MODE")
DIFFUSION_CKPT=$(get_config "DATASET_DIFFUSION_CKPT")
DDIM_STEPS=$(get_config "DATASET_DDIM_STEPS")
BATCH_SIZE=$(get_config "DATASET_BATCH_SIZE")

# --- Problem Settings ---
NUM_NODES=$(get_config "NUM_NODES")
TARGET_VEHICLES=$(get_config "TARGET_VEHICLES")
SEED=$(get_config "SEED")
DEVICE=$(get_config "DEVICE")
TOTAL_DEMAND=$(get_config "TOTAL_DEMAND")
MAX_C=$(get_config "MAX_C")

# =============================================================================
# Display Configuration
# =============================================================================

echo "=========================================="
echo "Dataset Generation Configuration"
echo "=========================================="
echo ""
echo "  DATASET:"
echo "    Output directory:   $OUTPUT_DIR"
echo "    Total episodes:     $TOTAL_EPISODES"
echo "    Test ratio:         $TEST_RATIO"
echo "    Generation mode:    $MODE"
echo ""
echo "  PROBLEM SETTINGS:"
echo "    Num nodes:          $NUM_NODES"
echo "    Target vehicles:    $TARGET_VEHICLES"
echo "    Random seed:        $SEED"
echo ""
if [[ "$MODE" == "diffusion" ]]; then
    echo "  DIFFUSION SETTINGS:"
    echo "    Checkpoint:         $DIFFUSION_CKPT"
    echo "    DDIM steps:         $DDIM_STEPS"
    echo "    Batch size:         $BATCH_SIZE"
    echo ""
fi
echo "  HARDWARE:"
echo "    Device:             $DEVICE"
echo ""

# Validate diffusion checkpoint if using diffusion mode
if [[ "$MODE" == "diffusion" ]]; then
    if [[ -z "$DIFFUSION_CKPT" ]]; then
        echo "ERROR: DATASET_DIFFUSION_CKPT must be set for diffusion mode"
        exit 1
    fi
    if [[ ! -f "$DIFFUSION_CKPT" ]]; then
        echo "ERROR: Diffusion checkpoint not found: $DIFFUSION_CKPT"
        exit 1
    fi
fi

# =============================================================================
# Build Command
# =============================================================================

cmd=(
    python3 -m adversarial_v2.generate_dataset
    --mode "$MODE"
    --total-episodes "$TOTAL_EPISODES"
    --test-ratio "$TEST_RATIO"
    --num-nodes "$NUM_NODES"
    --target-vehicles "$TARGET_VEHICLES"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --seed "$SEED"
)

if [[ "$MODE" == "diffusion" ]]; then
    cmd+=(
        --diffusion-checkpoint "$DIFFUSION_CKPT"
        --use-ddim
        --ddim-steps "$DDIM_STEPS"
        --batch-size "$BATCH_SIZE"
        --total-demand "$TOTAL_DEMAND"
        --max-c "$MAX_C"
    )
fi

echo "Generating dataset..."
echo ""
"${cmd[@]}"

echo ""
echo "=========================================="
echo "Dataset generation complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  Training data: $OUTPUT_DIR/train.pt"
echo "  Test data:     $OUTPUT_DIR/test.pt"
echo ""
echo "To use this dataset for training, set in static_config.py:"
echo "  TRAIN_DATA_PATH = \"$OUTPUT_DIR/train.pt\""
echo ""
