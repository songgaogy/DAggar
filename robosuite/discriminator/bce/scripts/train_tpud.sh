#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

SEED="${SEED:-42}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-${ROOT}/checkpoints/multitask_6/tpud}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="tpud_${TIMESTAMP}"
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

if [[ ! -f "${CKPT}" ]]; then
  echo "[train_tpud] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="offline"
export WANDB_ENTITY="songgao-personal"

echo "[train_tpud] ROOT=${ROOT}"
echo "[train_tpud] policy.ckpt=${CKPT}"
echo "[train_tpud] save_dir=${SAVE_DIR}"

cd "${ROOT}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/bce/train.py" \
  seed="${SEED}" \
  hydra.run.dir="${SAVE_DIR}" \
  save_name="${SAVE_NAME}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${CKPT}" \
  policy.encoder_batch_size="${ENCODER_BATCH_SIZE}" \
  data.image_size="${IMAGE_SIZE}" \
  data.transition_horizon="${HORIZON}" \
  training.batch_size="${BATCH_SIZE}" \
  training.num_workers="${NUM_WORKERS}" \
  training.epochs="${EPOCHS}" \
  training.lr="${LR}" \
  "$@"
