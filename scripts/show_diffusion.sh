#!/bin/bash

# =============================================================================
# Show diffusion model visualization
# Usage: bash scripts/show_diffusion.sh
#
# Configuration is read from static_config.py
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# =============================================================================
# Read configuration from static_config.py
# =============================================================================
get_config() {
    python3 -c "from static_config import $1; print($1)"
}

# --- Environment Settings ---
MAP_SIZE=$(get_config "MAP_SIZE")
NUM_SAMPLES=$(get_config "NUM_SAMPLES")
TOTAL_DEMAND=$(get_config "TOTAL_DEMAND")
MAX_C=$(get_config "MAX_C")

# --- Checkpoint ---
SHOW_DIFFUSION_CHECKPOINT=$(get_config "SHOW_DIFFUSION_CHECKPOINT")

# --- Output Settings ---
VIZ_SAVE_PATH=$(get_config "VIZ_SAVE_PATH")

echo "=========================================="
echo "Showing Diffusion Model Visualization:"
echo "  Environment:"
echo "    Map size: ${MAP_SIZE}x${MAP_SIZE}"
echo "    Num samples: $NUM_SAMPLES"
echo "    Total demand: $TOTAL_DEMAND"
echo "    Max capacity: $MAX_C"
echo "  Model:"
echo "    Checkpoint: ${SHOW_DIFFUSION_CHECKPOINT:-none}"
echo "  Output:"
echo "    Save path: $VIZ_SAVE_PATH"
echo "=========================================="

# Create output directory
mkdir -p "$(dirname "$VIZ_SAVE_PATH")"

# Build arguments
PYTHON_ARGS=()

PYTHON_ARGS+=(--checkpoint "$SHOW_DIFFUSION_CHECKPOINT")
PYTHON_ARGS+=(--map_size "$MAP_SIZE")
PYTHON_ARGS+=(--num_samples "$NUM_SAMPLES")
PYTHON_ARGS+=(--total-demand "$TOTAL_DEMAND")
PYTHON_ARGS+=(--max-c "$MAX_C")

if [[ -n "$VIZ_SAVE_PATH" ]]; then
    PYTHON_ARGS+=(--save_path "$VIZ_SAVE_PATH")
fi

python3 utils/visualize_diffusion.py "${PYTHON_ARGS[@]}"
