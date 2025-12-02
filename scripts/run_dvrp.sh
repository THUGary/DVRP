#!/bin/bash
# Run DVRP with V2 static model planner
# Usage: bash scripts/run_dvrp.sh [--render] [--seed N]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Default parameters
# Set STATIC_CKPT="" to use rule-based planner, or set a path to use model planner
STATIC_CKPT="${STATIC_CKPT:-}"
#
NUM_AGENTS=2
SEED=2025
RENDER="--render"
STATIC_DEMANDS="--static-demands"
RULE_MODE=""
# By default enable saving run outputs; set to empty string to disable
SAVE_RUN="--save-run"
# MAP_SIZE: Side length of the square map (map is MAP_SIZE × MAP_SIZE)
MAP_SIZE="${MAP_SIZE:-20}"
# TOTAL_DEMAND: Upper limit of sum of all customer demands (NOT node count!)
TOTAL_DEMAND="${TOTAL_DEMAND:-60}"
# NUM_NODES: Number of demand nodes
NUM_NODES="${NUM_NODES:-20}"
# MAX_STEPS = Maximum episode steps (default: unlimited)
MAX_STEPS=${MAX_STEPS:-}
# STATIC_MAX_END = Max environment time for static demands (default: 2 * max_time = 200)
STATIC_MAX_END=${STATIC_MAX_END:-}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --render)
            RENDER="--render"
            shift
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --num-agents)
            NUM_AGENTS="$2"
            shift 2
            ;;
        --dynamic)
            STATIC_DEMANDS=""
            shift
            ;;
        --ckpt)
            STATIC_CKPT="$2"
            shift 2
            ;;
        --rule-mode)
            RULE_MODE="$2"
            shift 2
            ;;
        --map-size)
            MAP_SIZE="$2"
            shift 2
            ;;
        --total-demand)
            TOTAL_DEMAND="$2"
            shift 2
            ;;
        --num-nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --save-run)
            SAVE_RUN="--save-run"
            shift
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --static-max-end)
            STATIC_MAX_END="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/run_dvrp.sh [--render] [--seed N] [--num-agents N] [--dynamic] [--ckpt PATH] [--map-size N] [--total-demand N] [--num-nodes N]"
            exit 1
            ;;
    esac
done

echo "Running DVRP with:"
echo "  Static checkpoint: $STATIC_CKPT"
echo "  Num agents: $NUM_AGENTS"
echo "  Seed: $SEED"
echo "  Render: ${RENDER:-disabled}"
echo "  Mode: ${STATIC_DEMANDS:+static}${STATIC_DEMANDS:-dynamic}"
echo "  Rule mode: ${RULE_MODE:-default}"
echo "  Map size: ${MAP_SIZE}x${MAP_SIZE}"
echo "  Num nodes: ${NUM_NODES}"
echo "  Total demand: ${TOTAL_DEMAND}"
echo "  Max steps: ${MAX_STEPS:-unlimited}"
echo "  Static max end: ${STATIC_MAX_END:-default (2*max_time)}"

# Note: do not auto-create `outputs/run` here; saving is controlled by run.py's --save-run flag


PYTHON_ARGS=()
if [[ -n "$STATIC_CKPT" ]]; then
    PYTHON_ARGS+=(--static-ckpt "$STATIC_CKPT")
fi
if [[ -n "$RULE_MODE" ]]; then
    PYTHON_ARGS+=(--rule-mode "$RULE_MODE")
fi
if [[ -n "$MAP_SIZE" ]]; then
    PYTHON_ARGS+=(--map-size "$MAP_SIZE")
fi
if [[ -n "$TOTAL_DEMAND" ]]; then
    PYTHON_ARGS+=(--total-demand "$TOTAL_DEMAND")
fi
if [[ -n "$NUM_NODES" ]]; then
    PYTHON_ARGS+=(--num-nodes "$NUM_NODES")
fi
if [[ -n "$MAX_STEPS" ]]; then
    PYTHON_ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$STATIC_MAX_END" ]]; then
    PYTHON_ARGS+=(--static-max-end "$STATIC_MAX_END")
fi

python3 run.py \
    --num-agents "$NUM_AGENTS" \
    --seed "$SEED" \
    $STATIC_DEMANDS \
    $RENDER \
    $SAVE_RUN \
    "${PYTHON_ARGS[@]}"
