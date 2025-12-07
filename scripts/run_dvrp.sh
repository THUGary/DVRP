#!/bin/bash

# =============================================================================
# Run DVRP with V2 static model planner
# Usage: bash scripts/run_dvrp.sh
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

# --- Planner Settings ---
STATIC_CKPT=$(get_config "STATIC_CKPT")
RULE_MODE=$(get_config "RULE_MODE")

# --- Environment Settings ---
NUM_AGENTS=$(get_config "NUM_AGENTS")
MAP_SIZE=$(get_config "MAP_SIZE")
NUM_NODES=$(get_config "NUM_NODES")
TOTAL_DEMAND=$(get_config "TOTAL_DEMAND")
MAX_TIME=$(get_config "MAX_TIME")

# --- Episode Settings ---
SEED=$(get_config "SEED")
STATIC_DEMANDS=$(get_config "STATIC_DEMANDS")
RENDER=$(get_config "RENDER")
SAVE_RUN=$(get_config "SAVE_RUN")

# =============================================================================
# Process configuration values
# =============================================================================

# Convert Python bool to bash flags
if [[ "$STATIC_DEMANDS" == "True" ]]; then
    STATIC_DEMANDS_FLAG="--static-demands"
else
    STATIC_DEMANDS_FLAG=""
fi

if [[ "$RENDER" == "True" ]]; then
    RENDER_FLAG="--render"
else
    RENDER_FLAG=""
fi

if [[ "$SAVE_RUN" == "True" ]]; then
    SAVE_RUN_FLAG="--save-run"
else
    SAVE_RUN_FLAG=""
fi

echo "=========================================="
echo "Running DVRP with:"
echo "  Planner:"
echo "    Static checkpoint: ${STATIC_CKPT:-none (rule-based)}"
echo "    Rule mode: ${RULE_MODE:-default}"
echo "  Environment:"
echo "    Num agents: $NUM_AGENTS"
echo "    Map size: ${MAP_SIZE}x${MAP_SIZE}"
echo "    Num nodes: ${NUM_NODES}"
echo "    Total demand: ${TOTAL_DEMAND}"
echo "    Max time: ${MAX_TIME}"
echo "  Episode:"
echo "    Seed: $SEED"
echo "    Render: ${RENDER}"
echo "    Static demands: ${STATIC_DEMANDS}"
echo "=========================================="

# Build Python arguments
PYTHON_ARGS=()

PYTHON_ARGS+=(--num-agents "$NUM_AGENTS")
PYTHON_ARGS+=(--seed "$SEED")

if [[ -n "$MAP_SIZE" ]]; then
    PYTHON_ARGS+=(--map-size "$MAP_SIZE")
fi

if [[ -n "$TOTAL_DEMAND" ]]; then
    PYTHON_ARGS+=(--total-demand "$TOTAL_DEMAND")
fi

if [[ -n "$NUM_NODES" ]]; then
    PYTHON_ARGS+=(--num-nodes "$NUM_NODES")
fi

if [[ -n "$MAX_TIME" ]]; then
    PYTHON_ARGS+=(--max-time "$MAX_TIME")
fi

if [[ -n "$STATIC_CKPT" ]]; then
    PYTHON_ARGS+=(--static-ckpt "$STATIC_CKPT")
fi

if [[ -n "$RULE_MODE" ]]; then
    PYTHON_ARGS+=(--rule-mode "$RULE_MODE")
fi

python3 run.py \
    $STATIC_DEMANDS_FLAG \
    $RENDER_FLAG \
    $SAVE_RUN_FLAG \
    "${PYTHON_ARGS[@]}"
