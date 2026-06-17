#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dodo/Documents/DAggar/robosuite"
PYTHON_BIN="/home/dodo/miniconda3/envs/dagger/bin/python"

GPU=0
SEED=42
BASE_SAVE_DIR="./checkpoints/multitask_6/dyn_bce"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="dyn_bce_${TIMESTAMP}"
SAVE_NAME="${RUN_NAME}.pt"
SAVE_DIR="${BASE_SAVE_DIR}/${RUN_NAME}"
CKPT="./checkpoints/multitask_6/flow-20/flow_multi_ep0100_20260320_114720.pt"
WANDB_MODE_VALUE="offline"
WANDB_ENTITY="songgao-personal"

BATCH_SIZE=512
NUM_WORKERS=16
PREFETCH_FACTOR=4
EPOCHS=200
LR=2e-4
IMAGE_SIZE=128
HORIZON=2   # you can change and try
ENCODER_BATCH_SIZE=96
MAX_TRANSITIONS_PER_SHARD=32768
NUM_BUILD_WORKERS=8
ENCODE_DEMO_BATCH_SIZE=8
MAX_PENDING_RAW_DEMOS=16
MEMMAP_REBUILD=false

TRAIN_EXPERT=180
TRAIN_SUCCESS=180
TRAIN_FAIL=180
VAL_EXPERT=15
VAL_SUCCESS=15
VAL_FAIL=15
TEST_EXPERT=5
TEST_SUCCESS=5
TEST_FAIL=5

export CUDA_VISIBLE_DEVICES="${GPU}"
export WANDB_MODE="${WANDB_MODE_VALUE}"
export WANDB_ENTITY="${WANDB_ENTITY}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/dyn_bce/train.py" \
  seed="${SEED}" \
  hydra.run.dir="${SAVE_DIR}" \
  save_name="${SAVE_NAME}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${CKPT}" \
  policy.encoder_batch_size="${ENCODER_BATCH_SIZE}" \
  data.image_size="${IMAGE_SIZE}" \
  data.transition_horizon="${HORIZON}" \
  data.memmap.max_transitions_per_shard="${MAX_TRANSITIONS_PER_SHARD}" \
  data.memmap.num_build_workers="${NUM_BUILD_WORKERS}" \
  data.memmap.encode_demo_batch_size="${ENCODE_DEMO_BATCH_SIZE}" \
  data.memmap.max_pending_raw_demos="${MAX_PENDING_RAW_DEMOS}" \
  data.memmap.rebuild="${MEMMAP_REBUILD}" \
  data.splits.train.num_expert_traj="${TRAIN_EXPERT}" \
  data.splits.train.num_success_traj="${TRAIN_SUCCESS}" \
  data.splits.train.num_fail_traj="${TRAIN_FAIL}" \
  data.splits.eval.num_expert_traj="${VAL_EXPERT}" \
  data.splits.eval.num_success_traj="${VAL_SUCCESS}" \
  data.splits.eval.num_fail_traj="${VAL_FAIL}" \
  data.splits.test.num_expert_traj="${TEST_EXPERT}" \
  data.splits.test.num_success_traj="${TEST_SUCCESS}" \
  data.splits.test.num_fail_traj="${TEST_FAIL}" \
  training.batch_size="${BATCH_SIZE}" \
  training.num_workers="${NUM_WORKERS}" \
  training.prefetch_factor="${PREFETCH_FACTOR}" \
  training.epochs="${EPOCHS}" \
  training.lr="${LR}" \
  logging.mode="${WANDB_MODE_VALUE}" \
  logging.entity="${WANDB_ENTITY}" \
  "$@"
