#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/gy/Documents/pi05-openpi"
CONFIG_NAME="pi05_arx_finetune_single_task_rtc"
TASK_NAME="put_shrimp_in_pot"
DATA_REPO_ID="arx_a5/put_shrimp_in_pot_openpi"
EXP_NAME="${CONFIG_NAME}-${TASK_NAME}-$(date +%Y%m%d)_$(date +%H%M%S)-rtc"

export XLA_PYTHON_CLIENT_MEM_FRACTION=.9
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_ENTITY="songgao-personal"

export HF_LEROBOT_HOME=${REPO_ROOT}/data/training
cd "${REPO_ROOT}"

# run this if repo_id/config_name is changed
uv run scripts/compute_norm_stats.py \
    --config-name "${CONFIG_NAME}" \
    --data.repo-id "${DATA_REPO_ID}"

uv run scripts/train.py \
    "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --overwrite \
    --data.repo-id="${DATA_REPO_ID}"
