#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/gy/Documents/pi05-openpi"
CONFIG_NAME="pi05_arx_finetune_single_task"
EXP_NAME="arx_finetune_single_task-put_shrimp_in_pot-$(date +%Y%m%d)"

export XLA_PYTHON_CLIENT_MEM_FRACTION=.9
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_ENTITY="songgao-personal"

export HF_LEROBOT_HOME=${REPO_ROOT}/data/training
cd "${REPO_ROOT}"

uv run scripts/compute_norm_stats.py \
    --config-name "${CONFIG_NAME}"

uv run scripts/train.py \
    "${CONFIG_NAME}" \
    --exp-name="${EXP_NAME}" \
    --overwrite
