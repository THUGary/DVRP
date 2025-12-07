#!/usr/bin/env bash

# =============================================================================
# Evaluate Different Planners on Various Distributions
# Usage: bash scripts/evaluate_distributions.sh
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
NUM_AGENTS=$(get_config "NUM_AGENTS")
NUM_NODES=$(get_config "NUM_NODES")
TOTAL_DEMAND=$(get_config "TOTAL_DEMAND")
MAX_C=$(get_config "MAX_C")
CAPACITY=$(get_config "CAPACITY")
MAP_SIZE=$(get_config "MAP_SIZE")

# --- Evaluation Settings ---
NUM_RUNS=$(get_config "NUM_RUNS")
STATIC_DEMANDS=$(get_config "STATIC_DEMANDS")
MAX_TIME=$(get_config "MAX_TIME")
SEED=$(get_config "SEED")

# --- POMO Inference Parameters ---
POMO_SIZE=$(get_config "POMO_SIZE")
AUG_FACTOR=$(get_config "AUG_FACTOR")

# --- Planners to Evaluate ---
RULE_MODES=$(get_config "RULE_MODES")
GLOBAL_OPT_MODES=$(get_config "GLOBAL_OPT_MODES")
MODEL_CHECKPOINTS=$(get_config "MODEL_CHECKPOINTS")

# --- Diffusion Generator Checkpoints ---
DIFFUSION_CHECKPOINTS=$(get_config "DIFFUSION_CHECKPOINTS")

# --- Problem Bank ---
PROBLEM_BANK_IN=$(get_config "PROBLEM_BANK_IN")
PROBLEM_BANK_OUT=$(get_config "PROBLEM_BANK_OUT")
GENERATE_ONLY=$(get_config "GENERATE_ONLY")

# --- Output Settings ---
OUT_DIR=$(get_config "OUT_DIR")
PLOT_METRICS=$(get_config "PLOT_METRICS")

# =============================================================================
# Process configuration values
# =============================================================================

# Process comma-separated values
RULE_MODES="${RULE_MODES//,/ }"
RULE_MODES="$(echo "$RULE_MODES" | xargs)"

GLOBAL_OPT_MODES="${GLOBAL_OPT_MODES//,/ }"
GLOBAL_OPT_MODES="$(echo "$GLOBAL_OPT_MODES" | xargs)"

MODEL_CHECKPOINTS="${MODEL_CHECKPOINTS//,/ }"
MODEL_CHECKPOINTS="$(echo "$MODEL_CHECKPOINTS" | xargs)"

DIFFUSION_CHECKPOINTS="${DIFFUSION_CHECKPOINTS//,/ }"
DIFFUSION_CHECKPOINTS="$(echo "$DIFFUSION_CHECKPOINTS" | xargs)"

# Convert Python bool to bash
if [[ "$STATIC_DEMANDS" == "True" ]]; then
    STATIC_DEMANDS="true"
else
    STATIC_DEMANDS="false"
fi

if [[ "$GENERATE_ONLY" == "True" ]]; then
    GENERATE_ONLY="true"
else
    GENERATE_ONLY="false"
fi

echo "=========================================="
echo "Evaluate Distributions Configuration"
echo "=========================================="
echo ""
echo "  ENVIRONMENT:"
echo "    Num agents:         $NUM_AGENTS"
echo "    Num nodes:          $NUM_NODES"
echo "    Total demand:       $TOTAL_DEMAND"
echo "    Max C:              $MAX_C"
echo "    Capacity:           $CAPACITY"
echo "    Map size:           ${MAP_SIZE}x${MAP_SIZE}"
echo ""
echo "  EVALUATION:"
echo "    Num runs:           $NUM_RUNS"
echo "    POMO size:          $POMO_SIZE"
echo "    Aug factor:         $AUG_FACTOR"
echo "    Max time:           ${MAX_TIME}"
echo "    Seed:               $SEED"
echo ""

RULES=()
if [[ -n "$RULE_MODES" ]]; then
    read -ra RULES <<<"$RULE_MODES"
fi

GLOBAL_OPTS=()
if [[ -n "$GLOBAL_OPT_MODES" ]]; then
    read -ra GLOBAL_OPTS <<<"$GLOBAL_OPT_MODES"
fi

DIFFUSION_ENTRIES=()
if [[ -n "$DIFFUSION_CHECKPOINTS" ]]; then
    read -ra DIFFUSION_ENTRIES <<<"$DIFFUSION_CHECKPOINTS"
fi

MODEL_ENTRIES=()
if [[ -n "$MODEL_CHECKPOINTS" ]]; then
    read -ra MODEL_ENTRIES <<<"$MODEL_CHECKPOINTS"
fi

# Validate model checkpoint files
for ckpt in "${MODEL_ENTRIES[@]}"; do
    if [[ "$ckpt" == *"="* ]]; then
        ckpt_path="${ckpt#*=}"
    else
        ckpt_path="$ckpt"
    fi
    if [[ ! -f "$ckpt_path" ]]; then
        echo "ERROR: Model checkpoint not found: $ckpt_path"
        exit 1
    fi
done

# Validate diffusion checkpoint files
for ckpt in "${DIFFUSION_ENTRIES[@]}"; do
    if [[ "$ckpt" == *"="* ]]; then
        ckpt_path="${ckpt#*=}"
    else
        ckpt_path="$ckpt"
    fi
    if [[ ! -f "$ckpt_path" ]]; then
        echo "ERROR: Diffusion checkpoint not found: $ckpt_path"
        exit 1
    fi
done

# Validate problem bank
if [[ -n "$PROBLEM_BANK_IN" && ! -f "$PROBLEM_BANK_IN" ]]; then
    echo "ERROR: Problem bank not found: $PROBLEM_BANK_IN"
    exit 1
fi

if [[ "${GENERATE_ONLY}" == "true" && -z "$PROBLEM_BANK_OUT" ]]; then
    echo "ERROR: GENERATE_ONLY requires PROBLEM_BANK_OUT"
    exit 1
fi

# =============================================================================
# Build Command
# =============================================================================

cmd=(
    python3 evaluate_distributions.py
    --seed "$SEED"
    --num-runs "$NUM_RUNS"
    --out-dir "$OUT_DIR"
    --plot-metrics "$PLOT_METRICS"
    --pomo-size "$POMO_SIZE"
    --aug-factor "$AUG_FACTOR"
)

if [[ -n "$NUM_AGENTS" ]]; then
    cmd+=(--num-agents "$NUM_AGENTS")
fi

if [[ -n "$NUM_NODES" ]]; then
    cmd+=(--num-nodes "$NUM_NODES")
fi

if [[ -n "$TOTAL_DEMAND" ]]; then
    cmd+=(--total-demand "$TOTAL_DEMAND")
fi

if [[ -n "$MAP_SIZE" ]]; then
    cmd+=(--map-size "$MAP_SIZE")
fi

if [[ -n "$CAPACITY" ]]; then
    cmd+=(--capacity "$CAPACITY")
fi

if [[ -n "$MAX_C" ]]; then
    cmd+=(--max-c "$MAX_C")
fi

if [[ -n "$MAX_TIME" ]]; then
    cmd+=(--max-time "$MAX_TIME")
fi

if [[ "${STATIC_DEMANDS}" == "true" ]]; then
    cmd+=(--static-demands)
fi

for mode in "${RULES[@]}"; do
    cmd+=(--rule-based "${mode}")
done

for mode in "${GLOBAL_OPTS[@]}"; do
    cmd+=(--global-opt "${mode}")
done

if [[ ${#MODEL_ENTRIES[@]} -gt 0 ]]; then
    cmd+=(--model-checkpoints)
    cmd+=("${MODEL_ENTRIES[@]}")
fi

if [[ ${#DIFFUSION_ENTRIES[@]} -gt 0 ]]; then
    cmd+=(--diffusion-checkpoints)
    cmd+=("${DIFFUSION_ENTRIES[@]}")
fi

if [[ -n "$PROBLEM_BANK_IN" ]]; then
    cmd+=(--problem-bank-in "$PROBLEM_BANK_IN")
fi

if [[ -n "$PROBLEM_BANK_OUT" ]]; then
    cmd+=(--problem-bank-out "$PROBLEM_BANK_OUT")
fi

if [[ "${GENERATE_ONLY}" == "true" ]]; then
    cmd+=(--generate-only)
fi

"${cmd[@]}"
