#!/usr/bin/env bash
# Evaluate Different Planners on Various Distributions
#
# KEY PARAMETERS:
#   - NUM_NODES: Number of demand nodes
#   - TOTAL_DEMAND: Upper limit of total demand to distribute (sum of all node demands)
#   - MAX_C: Max demand per node (default=5, range 1-5 per node)
#   - MAP_SIZE: Side length of the square map (map is MAP_SIZE × MAP_SIZE)
#   - NUM_AGENTS: Number of vehicles (default=2)
#
# FIXED PARAMETERS (match model training):
#   - Vehicle capacity: 30 (normalized to 1.0 for model)
#
# Usage Examples:
#   # Basic evaluation with default settings
#   bash scripts/evaluate_distributions.sh
#
#   # Evaluate with more nodes on larger map
#   NUM_NODES=50 MAP_SIZE=40 bash scripts/evaluate_distributions.sh
#
#   # Evaluate specific model checkpoint
#   MODEL_CHECKPOINTS=checkpoints/static_vrp_v2/best_n50.pt bash scripts/evaluate_distributions.sh

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# === Configurable Parameters ===
NUM_AGENTS="${NUM_AGENTS:-10}"             # Number of vehicles
# NUM_NODES: Number of demand nodes
NUM_NODES="${NUM_NODES:-20}"
# TOTAL_DEMAND: Upper limit of sum of all customer demands (NOT node count!)
TOTAL_DEMAND="${TOTAL_DEMAND:-80}"         # Total demand upper limit
MAP_SIZE="${MAP_SIZE:-50}"                 # Square map side length (map is MAP_SIZE × MAP_SIZE)

# POMO inference parameters
POMO_SIZE="${POMO_SIZE:-20}"              # Number of parallel rollouts
AUG_FACTOR="${AUG_FACTOR:-8}"             # Data augmentation factor

# Evaluation specific
RULE_MODES="${RULE_MODES:-optimize,greedy,heuristic}"
#optimize,greedy,exact,heuristic
RULE_MODES="${RULE_MODES//,/ }"
RULE_MODES="$(echo "$RULE_MODES" | xargs)"

GLOBAL_OPT_MODES="${GLOBAL_OPT_MODES:-}"
GLOBAL_OPT_MODES="${GLOBAL_OPT_MODES//,/ }"
GLOBAL_OPT_MODES="$(echo "$GLOBAL_OPT_MODES" | xargs)"

# Model checkpoints
MODEL_CHECKPOINTS="${MODEL_CHECKPOINTS:-checkpoints/static_vrp_v2/best_n80.pt}"
MODEL_CHECKPOINTS="${MODEL_CHECKPOINTS//,/ }"
MODEL_CHECKPOINTS="$(echo "$MODEL_CHECKPOINTS" | xargs)"

STATIC_DEMANDS="${STATIC_DEMANDS:-true}"
# Capacity and max demand per node (match model training)
CAPACITY="${CAPACITY:-30}"               # Vehicle capacity (fixed for model)
MAX_C="${MAX_C:-5}"                       # Max demand per node (1 to max_c)
OUT_DIR="${OUT_DIR:-outputs/eval}"
NUM_RUNS="${NUM_RUNS:-50}"
# Use inference_time_total for fair comparison (greedy/optimize call plan() many times,
# model calls once but should still compare total computation time)
PLOT_METRICS="${PLOT_METRICS:-failure_rate,total_distance,inference_time_total}"
# Increased limits to prevent cluster distribution failures
STATIC_MAX_END="${STATIC_MAX_END:-5000}"  # Time limit for static VRP
MAX_STEPS="${MAX_STEPS:-5000}"            # Max simulation steps

echo "=== Evaluate Distributions Configuration ==="
echo ""
echo "  DEMAND SETTINGS:"
echo "    Num nodes:          $NUM_NODES"
echo "    Total demand:       $TOTAL_DEMAND"
echo "    Max demand/node:    $MAX_C"
echo "    Vehicle capacity:   $CAPACITY"
echo ""
echo "  ENVIRONMENT:"
echo "    Num agents:         $NUM_AGENTS"
echo "    Map size:           ${MAP_SIZE}x${MAP_SIZE}"
echo ""
echo "  EVALUATION:"
echo "    Num runs:           $NUM_RUNS"
echo "    Rule modes:         $RULE_MODES"
echo "    Global opt:         $GLOBAL_OPT_MODES"
echo "    Model ckpts:        $MODEL_CHECKPOINTS"
echo "    POMO size:          $POMO_SIZE"
echo "    Aug factor:         $AUG_FACTOR"
echo "    Max steps:          ${MAX_STEPS:-<unlimited>}"
echo "    Static max end:     ${STATIC_MAX_END:-default (2*max_time)}"
echo ""

RULES=()
if [[ -n "$RULE_MODES" ]]; then
    read -ra RULES <<<"$RULE_MODES"
fi

GLOBAL_OPTS=()
if [[ -n "$GLOBAL_OPT_MODES" ]]; then
    read -ra GLOBAL_OPTS <<<"$GLOBAL_OPT_MODES"
fi

MODEL_ENTRIES=()
if [[ -n "$MODEL_CHECKPOINTS" ]]; then
    read -ra MODEL_ENTRIES <<<"$MODEL_CHECKPOINTS"
fi

# Validate all checkpoint files exist before running
for ckpt in "${MODEL_ENTRIES[@]}"; do
    # Handle label=path format
    if [[ "$ckpt" == *"="* ]]; then
        ckpt_path="${ckpt#*=}"
    else
        ckpt_path="$ckpt"
    fi
    if [[ ! -f "$ckpt_path" ]]; then
        echo "ERROR: Checkpoint file not found: $ckpt_path"
        exit 1
    fi
done

cmd+=(
    python3 evaluate_distributions.py
    --num-runs "$NUM_RUNS"
    --num-agents "$NUM_AGENTS"
    --num-nodes "$NUM_NODES"
    --total-demand "$TOTAL_DEMAND"
    --map-size "$MAP_SIZE"
    --out-dir "$OUT_DIR"
    --plot-metrics "$PLOT_METRICS"
    --pomo-size "$POMO_SIZE"
    --aug-factor "$AUG_FACTOR"
)

if [[ "${STATIC_DEMANDS:-false}" == "true" ]]; then
    cmd+=(--static-demands)
    if [[ -n "${STATIC_MAX_END:-}" ]]; then
        cmd+=(--static-max-end "${STATIC_MAX_END}")
    fi
fi

for mode in "${RULES[@]}"; do
    cmd+=(--rule-based "${mode}")
done

for mode in "${GLOBAL_OPTS[@]}"; do
    cmd+=(--global-opt "${mode}")
done

cmd+=(--model-checkpoints)
cmd+=("${MODEL_ENTRIES[@]}")

if [[ -n "$CAPACITY" ]]; then
    cmd+=(--capacity "$CAPACITY")
fi

if [[ -n "$MAX_C" ]]; then
    cmd+=(--max-c "$MAX_C")
fi

if [[ -n "$MAX_STEPS" ]]; then
    cmd+=(--max-steps "$MAX_STEPS")
fi

cmd+=("$@")

"${cmd[@]}"
