#!/bin/bash
# =============================================================================
# Run DVRP with V2 static model planner
# Usage: bash scripts/run_dvrp.sh
# 
# Edit the configuration variables below to change settings.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# =============================================================================
# CONFIGURATION - Edit these variables to change settings
# =============================================================================

# Model checkpoint path (set to "" to use rule-based planner)
STATIC_CKPT="checkpoints/cotrain/static_20251202_034046/planner_cycle_1.pt"

# Number of agents/vehicles
NUM_AGENTS=5

# Random seed for reproducibility
SEED=2025

# Enable rendering (set to "" to disable)
RENDER="--render"

# Static demands mode (set to "" for dynamic mode)
STATIC_DEMANDS="--static-demands"

# Rule-based planner mode (only used when STATIC_CKPT is empty)
# Options: "greedy", "exact", "heuristic"
RULE_MODE=""

# Save run outputs (set to "" to disable)
SAVE_RUN="--save-run"

# MAP_SIZE: Side length of the square map (map is MAP_SIZE × MAP_SIZE)
MAP_SIZE=40

# TOTAL_DEMAND: Upper limit of sum of all customer demands (NOT node count!)
TOTAL_DEMAND=150

# NUM_NODES: Number of demand nodes
NUM_NODES=30

# MAX_STEPS: Maximum episode steps (leave empty for unlimited)
MAX_STEPS=""

# NOTE: For static VRP, time limits are not used - episode ends when all
# demands are served and all agents return to depot.
# STATIC_MAX_END is kept for backward compatibility but ignored in static mode.

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

echo "=========================================="
echo "Running DVRP with:"
echo "  Static checkpoint: ${STATIC_CKPT:-none (rule-based)}"
echo "  Num agents: $NUM_AGENTS"
echo "  Seed: $SEED"
echo "  Render: ${RENDER:+enabled}${RENDER:-disabled}"
echo "  Mode: ${STATIC_DEMANDS:+static}${STATIC_DEMANDS:-dynamic}"
echo "  Rule mode: ${RULE_MODE:-default}"
echo "  Map size: ${MAP_SIZE}x${MAP_SIZE}"
echo "  Num nodes: ${NUM_NODES}"
echo "  Total demand: ${TOTAL_DEMAND}"
echo "  Max steps: ${MAX_STEPS:-unlimited}"
echo "  Static max end: ${STATIC_MAX_END:-default}"
echo "=========================================="

# Build Python arguments
PYTHON_ARGS=()

PYTHON_ARGS+=(--num-agents "$NUM_AGENTS")
PYTHON_ARGS+=(--seed "$SEED")
PYTHON_ARGS+=(--map-size "$MAP_SIZE")
PYTHON_ARGS+=(--total-demand "$TOTAL_DEMAND")
PYTHON_ARGS+=(--num-nodes "$NUM_NODES")

[[ -n "$STATIC_CKPT" ]] && PYTHON_ARGS+=(--static-ckpt "$STATIC_CKPT")
[[ -n "$RULE_MODE" ]] && PYTHON_ARGS+=(--rule-mode "$RULE_MODE")
[[ -n "$MAX_STEPS" ]] && PYTHON_ARGS+=(--max-steps "$MAX_STEPS")
[[ -n "$STATIC_MAX_END" ]] && PYTHON_ARGS+=(--static-max-end "$STATIC_MAX_END")

python3 run.py \
    $STATIC_DEMANDS \
    $RENDER \
    $SAVE_RUN \
    "${PYTHON_ARGS[@]}"
