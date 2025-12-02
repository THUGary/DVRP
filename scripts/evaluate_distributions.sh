#!/usr/bin/env bash
# =============================================================================
# Evaluate Different Planners on Various Distributions
# Usage: bash scripts/evaluate_distributions.sh
#
# Edit the configuration variables below to change settings.
# =============================================================================

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

# =============================================================================
# CONFIGURATION - Edit these variables to change settings
# =============================================================================

# --- Agent Settings ---
NUM_AGENTS=10              # Number of vehicles

# --- Demand Settings ---
NUM_NODES=20               # Number of demand nodes
TOTAL_DEMAND=80            # Upper limit of sum of all customer demands (NOT node count!)
MAX_C=5                    # Max demand per node (demands 1 to max_c)
CAPACITY=30                # Vehicle capacity (fixed for model)

# --- Environment Settings ---
MAP_SIZE=50                # Square map side length (map is MAP_SIZE × MAP_SIZE)

# --- Evaluation Settings ---
NUM_RUNS=50                # Number of evaluation runs per distribution
STATIC_DEMANDS="true"      # Use static demands mode ("true" or "false")
STATIC_MAX_END=5000        # Time limit for static VRP
MAX_STEPS=5000             # Max simulation steps

# --- POMO Inference Parameters ---
POMO_SIZE=20               # Number of parallel rollouts
AUG_FACTOR=8               # Data augmentation factor

# --- Planners to Evaluate ---
# Rule-based modes (comma-separated): optimize, greedy, exact, heuristic
RULE_MODES="optimize,greedy,heuristic"

# Global optimization modes (comma-separated, leave empty to skip)
GLOBAL_OPT_MODES=""

# Model checkpoints (comma-separated, or "label=path" format)
MODEL_CHECKPOINTS="checkpoints/static_vrp_v2/best_n80.pt"

# --- Output Settings ---
OUT_DIR="outputs/eval"
# Metrics to plot (comma-separated)
PLOT_METRICS="failure_rate,total_distance,inference_time_total"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

# Process comma-separated values
RULE_MODES="${RULE_MODES//,/ }"
RULE_MODES="$(echo "$RULE_MODES" | xargs)"

GLOBAL_OPT_MODES="${GLOBAL_OPT_MODES//,/ }"
GLOBAL_OPT_MODES="$(echo "$GLOBAL_OPT_MODES" | xargs)"

MODEL_CHECKPOINTS="${MODEL_CHECKPOINTS//,/ }"
MODEL_CHECKPOINTS="$(echo "$MODEL_CHECKPOINTS" | xargs)"

echo "=========================================="
echo "Evaluate Distributions Configuration"
echo "=========================================="
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

"${cmd[@]}"
