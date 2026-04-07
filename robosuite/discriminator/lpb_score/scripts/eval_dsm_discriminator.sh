#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

SEED="${SEED:-42}"
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
DSM_CKPT="${DSM_CKPT:-${ROOT}/checkpoints/multitask_6/lpb_score/lpb_score_dsm.pt}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_score/eval}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-96}"
ACTION_HORIZON="${ACTION_HORIZON:--1}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[eval_dsm_discriminator] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

if [[ ! -f "${DSM_CKPT}" ]]; then
  echo "[eval_dsm_discriminator] Missing DSM checkpoint: ${DSM_CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo "[eval_dsm_discriminator] ROOT=${ROOT}"
echo "[eval_dsm_discriminator] policy.ckpt=${CKPT}"
echo "[eval_dsm_discriminator] model.dsm_ckpt=${DSM_CKPT}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_score/eval_dsm_discriminator.py" \
  seed="${SEED}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${CKPT}" \
  policy.encoder_batch_size="${ENCODER_BATCH_SIZE}" \
  data.image_size="${IMAGE_SIZE}" \
  model.dsm_ckpt="${DSM_CKPT}" \
  feature.action_horizon="${ACTION_HORIZON}" \
  "$@"
