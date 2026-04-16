#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_dipole-new-v3}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="lpb_dipole_dsm_${TIMESTAMP}"
SAVE_NAME="${RUN_NAME}.pt"
SAVE_DIR="${BASE_SAVE_DIR}/${RUN_NAME}"
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"

BATCH_SIZE=1024
NUM_WORKERS=16
NUM_POS=350
NUM_NEG=100
POSITIVE_RATIO=0.7
EPOCHS=50

LR="${LR:-2e-4}"
IMAGE_SIZE=128
HORIZON=10
DSM_WINDOW_SIZE="${DSM_WINDOW_SIZE:-${HORIZON}}"
ENCODER_BATCH_SIZE=128
STD_CLAMP_MIN="${STD_CLAMP_MIN:-0.05}"
NOISE_SCALE="${NOISE_SCALE:-0.08}"


if [[ ! -f "${CKPT}" ]]; then
  echo "[train_lpb_score_dsm] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[train_lpb_score_dsm] ROOT=${ROOT}"
echo "[train_lpb_score_dsm] policy.ckpt=${CKPT}"
echo "[train_lpb_score_dsm] save_dir=${SAVE_DIR}"
echo "[train_lpb_score_dsm] train.num_pos_traj=${NUM_POS}"
echo "[train_lpb_score_dsm] train.num_neg_traj=${NUM_NEG}"
echo "[train_lpb_score_dsm] dataset.window_size=${DSM_WINDOW_SIZE}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_score/train.py" \
  seed="${SEED}" \
  hydra.run.dir="${SAVE_DIR}" \
  save_name="${SAVE_NAME}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${CKPT}" \
  policy.encoder_batch_size="${ENCODER_BATCH_SIZE}" \
  data.image_size="${IMAGE_SIZE}" \
  dataset.window_size="${DSM_WINDOW_SIZE}" \
  model.std_clamp_min="${STD_CLAMP_MIN}" \
  model.noise_scale="${NOISE_SCALE}" \
  data.splits.train.num_pos_traj="${NUM_POS}" \
  data.splits.train.num_neg_traj="${NUM_NEG}" \
  training.batch_size="${BATCH_SIZE}" \
  training.num_workers="${NUM_WORKERS}" \
  training.epochs="${EPOCHS}" \
  training.lr="${LR}" \
  training.positive_ratio="${POSITIVE_RATIO}" \
  "$@"
