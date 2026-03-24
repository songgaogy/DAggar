#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_new}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="lpb_new_${TIMESTAMP}"
SAVE_NAME="${RUN_NAME}.pt"
SAVE_DIR="${BASE_SAVE_DIR}/${RUN_NAME}"
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"

BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-2e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
HORIZON="${HORIZON:-1}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-96}"

TRAIN_EXPERT="${TRAIN_EXPERT:-180}"
TRAIN_SUCCESS="${TRAIN_SUCCESS:-180}"
TRAIN_FAIL="${TRAIN_FAIL:-180}"
VAL_EXPERT="${VAL_EXPERT:-15}"
VAL_SUCCESS="${VAL_SUCCESS:-15}"
VAL_FAIL="${VAL_FAIL:-15}"
TEST_EXPERT="${TEST_EXPERT:-5}"
TEST_SUCCESS="${TEST_SUCCESS:-5}"
TEST_FAIL="${TEST_FAIL:-5}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[train_lpb_new] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[train_lpb_new] ROOT=${ROOT}"
echo "[train_lpb_new] policy.ckpt=${CKPT}"
echo "[train_lpb_new] save_dir=${SAVE_DIR}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_new/train.py" \
  seed="${SEED}" \
  hydra.run.dir="${SAVE_DIR}" \
  save_name="${SAVE_NAME}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${CKPT}" \
  policy.encoder_batch_size="${ENCODER_BATCH_SIZE}" \
  data.image_size="${IMAGE_SIZE}" \
  data.transition_horizon="${HORIZON}" \
  data.splits.train.num_expert_traj="${TRAIN_EXPERT}" \
  data.splits.train.num_success_traj="${TRAIN_SUCCESS}" \
  data.splits.train.num_fail_traj="${TRAIN_FAIL}" \
  data.splits.val.num_expert_traj="${VAL_EXPERT}" \
  data.splits.val.num_success_traj="${VAL_SUCCESS}" \
  data.splits.val.num_fail_traj="${VAL_FAIL}" \
  data.splits.test.num_expert_traj="${TEST_EXPERT}" \
  data.splits.test.num_success_traj="${TEST_SUCCESS}" \
  data.splits.test.num_fail_traj="${TEST_FAIL}" \
  training.batch_size="${BATCH_SIZE}" \
  training.num_workers="${NUM_WORKERS}" \
  training.epochs="${EPOCHS}" \
  training.lr="${LR}" \
  "$@"
