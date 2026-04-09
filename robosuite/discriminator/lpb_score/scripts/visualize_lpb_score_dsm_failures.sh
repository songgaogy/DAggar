#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

SEED=1
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
DSM_CKPT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/lpb_score/lpb_score_dsm_20260408_224851/lpb_score_dsm_20260408_224851_ep0030.pt"
SAVE_DIR="${SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_score/visualize}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/data/.lpb_new_cache}"
NUM_VIDEOS=8
DELTA_THRESHOLD=20.0

BANK_SIZE=100
WINDOW_SIZE=6

IMAGE_SIZE="${IMAGE_SIZE:-128}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-96}"
FPS="${FPS:-20}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
SAVE_PDF="${SAVE_PDF:-true}"
NUM_PLOT_FRAMES="${NUM_PLOT_FRAMES:-8}"
ACTION_HORIZON="${ACTION_HORIZON:--1}"
CALIBRATION_SEED="${CALIBRATION_SEED:-${SEED}}"

if [[ -z "${DSM_CKPT}" ]]; then
  mapfile -t DSM_CKPT_CANDIDATES < <(
    find "${ROOT}/checkpoints/multitask_6/lpb_score" -type f -name 'lpb_score_dsm_*.pt' ! -name '*_ep*.pt' | sort
  )
  if [[ "${#DSM_CKPT_CANDIDATES[@]}" -gt 0 ]]; then
    DSM_CKPT="${DSM_CKPT_CANDIDATES[-1]}"
  else
    DSM_CKPT="${ROOT}/checkpoints/multitask_6/lpb_score/lpb_score_dsm.pt"
  fi
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[visualize_lpb_score_dsm] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

if [[ ! -f "${DSM_CKPT}" ]]; then
  echo "[visualize_lpb_score_dsm] Missing DSM checkpoint: ${DSM_CKPT}" >&2
  echo "[visualize_lpb_score_dsm] Set DSM_CKPT=/abs/path/to/lpb_score_dsm_*.pt if you want a specific run." >&2
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
        f"[visualize_lpb_score_dsm] Invalid policy checkpoint: {path}\n"
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
        f"[visualize_lpb_score_dsm] Invalid DSM checkpoint: {path}\n"
        "Expected an lpb_score DSM checkpoint with keys like `model`, `latent_dim`, and `action_dim`."
    )
PY
}

validate_policy_ckpt "${CKPT}"
validate_dsm_ckpt "${DSM_CKPT}"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo "[visualize_lpb_score_dsm] ROOT=${ROOT}"
echo "[visualize_lpb_score_dsm] policy.ckpt=${CKPT}"
echo "[visualize_lpb_score_dsm] model.dsm_ckpt=${DSM_CKPT}"
echo "[visualize_lpb_score_dsm] data.cache_dir=${CACHE_DIR}"
echo "[visualize_lpb_score_dsm] visualization.bank_size=${BANK_SIZE}"
echo "[visualize_lpb_score_dsm] visualization.calibration_seed=${CALIBRATION_SEED}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_score/visualize_failures.py" \
  seed="${SEED}" \
  save_dir="${SAVE_DIR}" \
  data.cache_dir="${CACHE_DIR}" \
  policy.ckpt="${CKPT}" \
  policy.encoder_batch_size="${ENCODER_BATCH_SIZE}" \
  data.image_size="${IMAGE_SIZE}" \
  model.dsm_ckpt="${DSM_CKPT}" \
  feature.action_horizon="${ACTION_HORIZON}" \
  visualization.bank_size="${BANK_SIZE}" \
  visualization.calibration_seed="${CALIBRATION_SEED}" \
  visualization.num_videos="${NUM_VIDEOS}" \
  visualization.fps="${FPS}" \
  visualization.camera_name="${CAMERA_NAME}" \
  visualization.save_pdf="${SAVE_PDF}" \
  visualization.num_plot_frames="${NUM_PLOT_FRAMES}" \
  detector.lambda_window_size=$WINDOW_SIZE \
  detector.delta="${DELTA_THRESHOLD}" \
  "$@"
