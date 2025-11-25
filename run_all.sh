#!/usr/bin/env bash
set -euo pipefail

# 参数可按需修改
MAP_WID=${MAP_WID:-20}
MAP_HEI=${MAP_HEI:-20}
AGENTS=${AGENTS:-2}
EPISODES=${EPISODES:-200}
VAL_RATIO=${VAL_RATIO:-0.1}
DATA_DIR=${DATA_DIR:-data}
CKPT_DIR=${CKPT_DIR:-checkpoints/planner}
EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-256}
LR=${LR:-1e-3}
SEED=${SEED:-42}
REPLAN=${REPLAN:-always} # always | on_new_or_empty
PLAN_HORIZON=${PLAN_HORIZON:-8}
STATIC_DATA_DIR="${DATA_DIR}/static_rows"
DYNAMIC_DATA_DIR="${DATA_DIR}/dynamicrows"

echo "=== 1) Generating STATIC snapshot rows (episodes=$EPISODES, agents=$AGENTS, horizon=$PLAN_HORIZON) ==="
python training/planner/data_gen.py \
  --episodes "${EPISODES}" \
  --planner greedy \
  --map_wid "${MAP_WID}" \
  --map_hei "${MAP_HEI}" \
  --agent_num "${AGENTS}" \
  --seed "${SEED}" \
  --val_ratio "${VAL_RATIO}" \
  --out_dir "${STATIC_DATA_DIR}" \
  --replan_policy "${REPLAN}" \
  --plan_horizon "${PLAN_HORIZON}" \
  --stage static

echo "=== 2) Training static CVRP planner (epochs=$EPOCHS) ==="
python training/planner/train_model.py \
  --data_dir "${STATIC_DATA_DIR}" \
  --map_wid "${MAP_WID}" \
  --agent_num "${AGENTS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --lr "${LR}" \
  --device cuda \
  --stage static \
  --ckpt_dir "${CKPT_DIR}"

STATIC_CKPT=$(ls -t "${CKPT_DIR}"/planner_static_${MAP_WID}_${AGENTS}_*.pt 2>/dev/null | head -n 1)
if [ -z "$STATIC_CKPT" ]; then
  echo "ERROR: No static checkpoint found in ${CKPT_DIR}."
  exit 1
fi
echo "Static checkpoint: ${STATIC_CKPT}"

echo "=== 3) Generating DYNAMIC rows for adapter training ==="
python training/planner/data_gen.py \
  --episodes "${EPISODES}" \
  --planner greedy \
  --map_wid "${MAP_WID}" \
  --map_hei "${MAP_HEI}" \
  --agent_num "${AGENTS}" \
  --seed "${SEED}" \
  --val_ratio "${VAL_RATIO}" \
  --out_dir "${DYNAMIC_DATA_DIR}" \
  --replan_policy "${REPLAN}" \
  --plan_horizon "${PLAN_HORIZON}" \
  --stage dynamic

echo "=== 4) Training dynamic adapter (epochs=$EPOCHS) ==="
python training/planner/train_model.py \
  --data_dir "${DYNAMIC_DATA_DIR}" \
  --map_wid "${MAP_WID}" \
  --agent_num "${AGENTS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --lr "${LR}" \
  --device cuda \
  --stage dynamic \
  --static_ckpt "${STATIC_CKPT}" \
  --adapter_dim 64 \
  --ckpt_dir "${CKPT_DIR}"

CKPT_PATH=$(ls -t "${CKPT_DIR}"/planner_dynamic_${MAP_WID}_${AGENTS}_*.pt 2>/dev/null | head -n 1)
if [ -z "$CKPT_PATH" ]; then
  echo "WARNING: No dynamic checkpoint found. Using static checkpoint for evaluation."
  CKPT_PATH="$STATIC_CKPT"
fi
echo "Checkpoint: ${CKPT_PATH}"

echo "=== 3) Testing trained ModelPlanner in env (render off by default) ==="
echo python3 test_model.py \
  --ckpt "${CKPT_PATH}" \
  --map_wid "${MAP_WID}" \
  --map_hei "${MAP_HEI}" \
  --agent_num "${AGENTS}"
if [ -n "${CKPT_PATH}" ]; then
  python3 test_model.py \
    --ckpt "${CKPT_PATH}" \
    --map_wid "${MAP_WID}" \
    --map_hei "${MAP_HEI}" \
    --agent_num "${AGENTS}"
else
  echo "Skipping test_model.py because no checkpoint is available."
fi