#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

SEED=2
DSM_CKPT="${DSM_CKPT:-checkpoints/multitask_6/lpb_dipole-new-v3/lpb_dipole_dsm_20260417_005436/lpb_dipole_dsm_20260417_005436.pt}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_dipole-new-v4/visualize}"
NUM_VIDEOS=12

BANK_SIZE=100
VIS_DATA_SOURCE="fail_rollout"  # suboptimal | expert | success_rollout | fail_rollout
SCORE_MODE="t3_weighted_combo"   # only supported score

# ---------------------------------------
# chunk energy
ALPHA_STATE=1
ALPHA="${ALPHA:-${ALPHA_STATE}}"

# chunk margin
BETA_STATE=1
BETA="${BETA:-${BETA_STATE}}"

# threshold
# NOTE: 5-8% should be better
DELTA_THRESHOLD=8

LAMBDA_MODE="mean"  # mean | max
WINDOW_SIZE=10
DSM_WINDOW_SIZE="${DSM_WINDOW_SIZE:-10}"
# ---------------------------------------

FPS="${FPS:-20}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
SAVE_PDF="${SAVE_PDF:-true}"
NUM_PLOT_FRAMES="${NUM_PLOT_FRAMES:-8}"
CALIBRATION_SEED="${CALIBRATION_SEED:-${SEED}}"
PREPROCESSED_CACHE_DIR="${PREPROCESSED_CACHE_DIR:-${ROOT}/data/.lpb_score_preprocessed_cache}"
USE_PREPROCESSED_CACHE="${USE_PREPROCESSED_CACHE:-1}"
REFRESH_PREPROCESSED_CACHE="${REFRESH_PREPROCESSED_CACHE:-0}"

if [[ "${USE_PREPROCESSED_CACHE}" == "1" ]]; then
  USE_PREPROCESSED_CACHE_BOOL="true"
else
  USE_PREPROCESSED_CACHE_BOOL="false"
fi

if [[ "${REFRESH_PREPROCESSED_CACHE}" == "1" ]]; then
  REFRESH_PREPROCESSED_CACHE_BOOL="true"
else
  REFRESH_PREPROCESSED_CACHE_BOOL="false"
fi

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

if [[ ! -f "${DSM_CKPT}" ]]; then
  echo "[visualize_lpb_score_dsm] Missing DSM checkpoint: ${DSM_CKPT}" >&2
  echo "[visualize_lpb_score_dsm] Set DSM_CKPT=/abs/path/to/lpb_score_dsm_*.pt if you want a specific run." >&2
  exit 1
fi

validate_dsm_ckpt() {
  "${PYTHON_BIN}" - "${1}" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(path, map_location="cpu")

required = {"model", "latent_dim", "window_size", "policy_checkpoint_payload"}
missing = sorted(required.difference(payload.keys()))
if missing:
    raise SystemExit(
        f"[visualize_lpb_score_dsm] Invalid DSM checkpoint: {path}\n"
        "Expected a self-contained joint DSM checkpoint with encoder payload.\n"
        f"Missing keys: {missing}"
    )
PY
}

validate_dsm_ckpt "${DSM_CKPT}"

export CUDA_VISIBLE_DEVICES=1

echo "[visualize_lpb_score_dsm] ROOT=${ROOT}"
echo "[visualize_lpb_score_dsm] model.dsm_ckpt=${DSM_CKPT}"
echo "[visualize_lpb_score_dsm] visualization.data_source=${VIS_DATA_SOURCE}"
echo "[visualize_lpb_score_dsm] visualization.bank_size=${BANK_SIZE}"
echo "[visualize_lpb_score_dsm] visualization.calibration_seed=${CALIBRATION_SEED}"
echo "[visualize_lpb_score_dsm] dataset.window_size=${DSM_WINDOW_SIZE}"
echo "[visualize_lpb_score_dsm] detector.score_mode=${SCORE_MODE}"
echo "[visualize_lpb_score_dsm] detector.alpha=${ALPHA}"
echo "[visualize_lpb_score_dsm] detector.beta=${BETA}"
echo "[visualize_lpb_score_dsm] data.preprocessed_cache_dir=${PREPROCESSED_CACHE_DIR}"
echo "[visualize_lpb_score_dsm] data.use_preprocessed_cache=${USE_PREPROCESSED_CACHE_BOOL}"
echo "[visualize_lpb_score_dsm] data.refresh_preprocessed_cache=${REFRESH_PREPROCESSED_CACHE_BOOL}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_score/visualize_failures.py" \
  seed="${SEED}" \
  save_dir="${SAVE_DIR}" \
  model.dsm_ckpt="${DSM_CKPT}" \
  data.preprocessed_cache_dir="${PREPROCESSED_CACHE_DIR}" \
  data.use_preprocessed_cache="${USE_PREPROCESSED_CACHE_BOOL}" \
  data.refresh_preprocessed_cache="${REFRESH_PREPROCESSED_CACHE_BOOL}" \
  dataset.window_size="${DSM_WINDOW_SIZE}" \
  visualization.data_source="${VIS_DATA_SOURCE}" \
  visualization.bank_size="${BANK_SIZE}" \
  visualization.calibration_seed="${CALIBRATION_SEED}" \
  visualization.num_videos="${NUM_VIDEOS}" \
  visualization.fps="${FPS}" \
  visualization.camera_name="${CAMERA_NAME}" \
  visualization.save_pdf="${SAVE_PDF}" \
  visualization.num_plot_frames="${NUM_PLOT_FRAMES}" \
  detector.lambda_window_size=$WINDOW_SIZE \
  detector.delta="${DELTA_THRESHOLD}" \
  detector.alpha="${ALPHA}" \
  detector.beta="${BETA}" \
  "$@"
