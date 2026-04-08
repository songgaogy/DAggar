#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

SEED="${SEED:-42}"
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
DSM_CKPT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/lpb_score/analyse/lpb_score_dsm_threshold_20260408_230818.json"
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

validate_policy_ckpt() {
  "${PYTHON_BIN}" - "${1}" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(path, map_location="cpu")

if "camera_names" not in payload or "model_cfg" not in payload:
    raise SystemExit(
        f"[eval_dsm_discriminator] Invalid policy checkpoint: {path}\n"
        "Expected a flow policy checkpoint with keys like `camera_names` and `model_cfg`.\n"
        "It looks like you may have passed a DSM checkpoint as policy.ckpt."
    )
PY
}

validate_dsm_ckpt() {
  "${PYTHON_BIN}" - "${1}" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(path, map_location="cpu")

if "model" not in payload or "latent_dim" not in payload or "action_dim" not in payload:
    raise SystemExit(
        f"[eval_dsm_discriminator] Invalid DSM checkpoint: {path}\n"
        "Expected an lpb_score DSM checkpoint with keys like `model`, `latent_dim`, and `action_dim`."
    )
PY
}

validate_policy_ckpt "${CKPT}"
validate_dsm_ckpt "${DSM_CKPT}"

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
