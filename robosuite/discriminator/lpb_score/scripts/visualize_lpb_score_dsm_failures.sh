#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

SEED=2
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
# TODO: update DSM_CKPT after the SσDC refactor retrain; the old uni-dsm three-head checkpoint
# (checkpoints/multitask_6/lpb_dipole-new/lpb_dipole_dsm_20260412_002031) is no longer loadable
# because the action head has been dropped from the architecture.
DSM_CKPT="checkpoints/multitask_6/lpb_dipole-new-v6/lpb_dipole_dsm_20260420_070130/lpb_dipole_dsm_20260420_070130.pt"
SAVE_DIR="${SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_dipole-new-v6/visualize}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/data/.lpb_score_cache}"
# Set to 1 to encode missing .npz caches on the fly (needed if HDF5 mtimes changed or cache is partial).
BUILD_MISSING_CACHE=1
NUM_VIDEOS=12

BANK_SIZE=100
VIS_DATA_SOURCE="fail_rollout"  # suboptimal | expert | success_rollout | fail_rollout

# ---------------------------------------
# SσDC per-branch weights. Default (α=0, β=1) replicates the R5 recipe.
# α_b · z(E_b^+) is the OOD-prior term; β_b · (z(E_b^+) - z(E_b^-)) is the LLR term.
ALPHA_STATE=0
ALPHA_DYNAMICS=0

BETA_STATE=1
BETA_DYNAMICS=1

# threshold
# NOTE: 3-5% should be better
DELTA_THRESHOLD=5

LAMBDA_MODE="mean"   # mean | max (SσDC detector does not support ema)
WINDOW_SIZE=6
# ---------------------------------------

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

export CUDA_VISIBLE_DEVICES=1

echo "[visualize_lpb_score_dsm] ROOT=${ROOT}"
echo "[visualize_lpb_score_dsm] policy.ckpt=${CKPT}"
echo "[visualize_lpb_score_dsm] model.dsm_ckpt=${DSM_CKPT}"
echo "[visualize_lpb_score_dsm] data.cache_dir=${CACHE_DIR}"
echo "[visualize_lpb_score_dsm] data.build_missing_cache=${BUILD_MISSING_CACHE}"
echo "[visualize_lpb_score_dsm] visualization.data_source=${VIS_DATA_SOURCE}"
echo "[visualize_lpb_score_dsm] visualization.bank_size=${BANK_SIZE}"
echo "[visualize_lpb_score_dsm] visualization.calibration_seed=${CALIBRATION_SEED}"
echo "[visualize_lpb_score_dsm] detector.alpha=(state=${ALPHA_STATE}, dynamics=${ALPHA_DYNAMICS})"
echo "[visualize_lpb_score_dsm] detector.beta=(state=${BETA_STATE}, dynamics=${BETA_DYNAMICS})"
echo "[visualize_lpb_score_dsm] detector.lambda_mode=${LAMBDA_MODE}"

BUILD_MISSING_CACHE_ARG=()
if [[ "${BUILD_MISSING_CACHE}" == "1" || "${BUILD_MISSING_CACHE}" == "true" ]]; then
  BUILD_MISSING_CACHE_ARG=(data.build_missing_cache=true)
fi

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_score/visualize_failures.py" \
  seed="${SEED}" \
  save_dir="${SAVE_DIR}" \
  data.cache_dir="${CACHE_DIR}" \
  "${BUILD_MISSING_CACHE_ARG[@]}" \
  policy.ckpt="${CKPT}" \
  policy.encoder_batch_size="${ENCODER_BATCH_SIZE}" \
  data.image_size="${IMAGE_SIZE}" \
  model.dsm_ckpt="${DSM_CKPT}" \
  feature.action_horizon="${ACTION_HORIZON}" \
  visualization.data_source="${VIS_DATA_SOURCE}" \
  visualization.bank_size="${BANK_SIZE}" \
  visualization.calibration_seed="${CALIBRATION_SEED}" \
  visualization.num_videos="${NUM_VIDEOS}" \
  visualization.fps="${FPS}" \
  visualization.camera_name="${CAMERA_NAME}" \
  visualization.save_pdf="${SAVE_PDF}" \
  visualization.num_plot_frames="${NUM_PLOT_FRAMES}" \
  detector.lambda_mode="${LAMBDA_MODE}" \
  detector.lambda_window_size=$WINDOW_SIZE \
  detector.delta="${DELTA_THRESHOLD}" \
  detector.alpha_state="${ALPHA_STATE}" \
  detector.alpha_dynamics="${ALPHA_DYNAMICS}" \
  detector.beta_state="${BETA_STATE}" \
  detector.beta_dynamics="${BETA_DYNAMICS}" \
  "$@"
