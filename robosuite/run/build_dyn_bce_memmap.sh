#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dodo/Documents/DAggar/robosuite"
PYTHON_BIN="/home/dodo/miniconda3/envs/dagger/bin/python"

GPU=0
SEED=42
CKPT="./checkpoints/multitask_6/flow-20/flow_multi_ep0100_20260320_114720.pt"

IMAGE_SIZE=128
HORIZON=1
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

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/dyn_bce/build_memmap.py" \
  seed="${SEED}" \
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
  data.splits.val.num_expert_traj="${VAL_EXPERT}" \
  data.splits.val.num_success_traj="${VAL_SUCCESS}" \
  data.splits.val.num_fail_traj="${VAL_FAIL}" \
  data.splits.test.num_expert_traj="${TEST_EXPERT}" \
  data.splits.test.num_success_traj="${TEST_SUCCESS}" \
  data.splits.test.num_fail_traj="${TEST_FAIL}" \
  "$@"
