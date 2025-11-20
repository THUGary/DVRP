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

echo "=== 1) Generating data with planner (episodes=$EPISODES, agents=$AGENTS, map=${MAP_WID}x${MAP_HEI}) ==="
echo python3 data_gen.py \
  --episodes "${EPISODES}" \
  --planner greedy \
  --map_wid "${MAP_WID}" \
  --map_hei "${MAP_HEI}" \
  --agent_num "${AGENTS}" \
  --seed "${SEED}" \
  --val_ratio "${VAL_RATIO}" \
  --out_dir "${DATA_DIR}" \
  --replan_policy "${REPLAN}"

python3 data_gen.py \
  --episodes "${EPISODES}" \
  --planner greedy \
  --map_wid "${MAP_WID}" \
  --map_hei "${MAP_HEI}" \
  --agent_num "${AGENTS}" \
  --seed "${SEED}" \
  --val_ratio "${VAL_RATIO}" \
  --out_dir "${DATA_DIR}" \
  --replan_policy "${REPLAN}"

echo "=== 2) Training DVRPNet on generated rows (epochs=$EPOCHS) ==="
echo python3 train_model.py \
  --data_dir "${DATA_DIR}" \
  --map_wid "${MAP_WID}" \
  --agent_num "${AGENTS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --lr "${LR}" \
  --device cuda
python3 train_model.py \
  --data_dir "${DATA_DIR}" \
  --map_wid "${MAP_WID}" \
  --agent_num "${AGENTS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH}" \
  --lr "${LR}" \
  --device cuda
# python3 train_model.py \
#   --data_dir "${DATA_DIR}" \
#   --map_wid "${MAP_WID}" \
#   --agent_num "${AGENTS}" \
#   --epochs "${EPOCHS}" \
#   --batch_size "${BATCH}" \
#   --lr "${LR}" \
#   --device cpu

CKPT_PATH="${CKPT_DIR}/planner_${MAP_WID}_${AGENTS}_${EPOCHS}.pt"
if [ ! -f "${CKPT_PATH}" ]; then
  echo "WARNING: expected checkpoint not found at ${CKPT_PATH}."
  files=( "${CKPT_DIR}"/planner_"${MAP_WID}"_"${AGENTS}"_*.pt )
  if [ -e "${files[0]}" ]; then
    CKPT_PATH="${files[0]}"
    echo "Using latest checkpoint: ${CKPT_PATH}"
  else
    echo "No checkpoint files found in ${CKPT_DIR}." 
    CKPT_PATH=""
  fi
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