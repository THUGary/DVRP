#!/bin/bash
# Run DVRP with V2 static model planner
# Usage: bash scripts/run_dvrp.sh [--render] [--seed N]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Default parameters
# Set STATIC_CKPT="" to use rule-based planner, or set a path to use model planner
STATIC_CKPT="/home/user0/DVRP-11.23/checkpoints/static_vrp_v2/best_n80.pt"
#
NUM_AGENTS=10
SEED=2025
RENDER="--render"
STATIC_DEMANDS="--static-demands"
RULE_MODE=""
# By default enable saving run outputs; set to empty string to disable
SAVE_RUN="--save-run"
MAP_WIDTH=50
MAP_HEIGHT=50
# PROBLEM_SIZE = TOTAL_DEMAND = Number of customer nodes to visit
PROBLEM_SIZE=${PROBLEM_SIZE:-${TOTAL_DEMAND:-100}}
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
        --map-wid)
            MAP_WIDTH="$2"
            shift 2
            ;;
        --map-hei)
            MAP_HEIGHT="$2"
            shift 2
            ;;
        --total-demand|--problem-size)
            PROBLEM_SIZE="$2"
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
            echo "Usage: bash scripts/run_dvrp.sh [--render] [--seed N] [--num-agents N] [--dynamic] [--ckpt PATH]"
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
echo "  Map size: ${MAP_WIDTH:-default}x${MAP_HEIGHT:-default}"
echo "  Problem size: ${PROBLEM_SIZE:-default} nodes"
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
if [[ -n "$MAP_WIDTH" ]]; then
    PYTHON_ARGS+=(--map-wid "$MAP_WIDTH")
fi
if [[ -n "$MAP_HEIGHT" ]]; then
    PYTHON_ARGS+=(--map-hei "$MAP_HEIGHT")
fi
if [[ -n "$PROBLEM_SIZE" ]]; then
    PYTHON_ARGS+=(--total-demand "$PROBLEM_SIZE")
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
