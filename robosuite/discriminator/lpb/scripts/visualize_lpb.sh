#!/usr/bin/env bash
# LPB KNN failure-detector visualization entry point.
# Run from repo root:
#   bash robosuite/discriminator/lpb/scripts/visualize_lpb.sh
# Overrides via env vars, e.g.:
#   TASK=PickPlaceCan NUM_TRAJS=6 SEED=42 USE_TRANSITION_ERROR=1 \
#     bash robosuite/discriminator/lpb/scripts/visualize_lpb.sh
#
# Required: LPB_CKPT must point to a pretrained LPB dynamics checkpoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/utils/fail_rollout}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/utils/success_rollout}"
TASK="${TASK:-PickPlaceCereal}"
NUM_TRAJS="${NUM_TRAJS:-3}"
SEED="${SEED:-0}"
FPS="${FPS:-20}"
BORDER_THICKNESS="${BORDER_THICKNESS:-10}"

# Per-task caps (control fit/scoring scale).
MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-100}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-200}"

RUN_NAME="${RUN_NAME:-viz_${TASK}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/lpb/viz/${RUN_NAME}}"
PDF_NAME="${PDF_NAME:-lpb_scores.pdf}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# Required checkpoint.
LPB_CKPT="${LPB_CKPT:-${REPO_ROOT}/checkpoints/lpb/dynamics/dynamics_model.pt}"
if [[ ! -f "${LPB_CKPT}" ]]; then
    echo "[lpb][viz] ERROR: LPB_CKPT not found: ${LPB_CKPT}" >&2
    echo "           Set LPB_CKPT=/abs/path/to/dynamics_model.pt (from train_lpb_dynamics.sh)." >&2
    exit 1
fi

# Feature extractor.
DEVICE="${DEVICE:-cuda}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-256}"
ACTION_HORIZON="${ACTION_HORIZON:--1}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
TRANSITION_PROPRIO_ERROR_WEIGHT="${TRANSITION_PROPRIO_ERROR_WEIGHT:-0.1}"

# Detector.
DELTA="${DELTA:-10.0}"
DELTA_STEP="${DELTA_STEP:-1.0}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-8192}"
LAMBDA_MODE="${LAMBDA_MODE:-mean}"
LAMBDA_WINDOW_SIZE="${LAMBDA_WINDOW_SIZE:--1}"
TRANSITION_AUX_WEIGHT="${TRANSITION_AUX_WEIGHT:-0.0}"

# Calibration / misc.
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"

EXTRA_ARGS=()
if [[ -n "${MAX_FAIL_PER_TASK}" && "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_SUCCESS_PER_TASK}" && "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi
if [[ "${USE_TRANSITION_ERROR:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--use-transition-error)
fi
if [[ "${NO_NORMALIZE_FEATURE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-normalize-feature)
fi
if [[ "${QUIET_FIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--quiet-fit)
fi
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    EXTRA_ARGS+=(--proprio-indices ${PROPRIO_INDICES})
fi

"${PYTHON_BIN}" -m robosuite.discriminator.lpb.visualize \
    --lpb-ckpt                        "${LPB_CKPT}" \
    --fail-root                       "${FAIL_ROOT}" \
    --success-root                    "${SUCCESS_ROOT}" \
    --task                            "${TASK}" \
    --num-trajs                       "${NUM_TRAJS}" \
    --out-dir                         "${OUT_DIR}" \
    --pdf-name                        "${PDF_NAME}" \
    --seed                            "${SEED}" \
    --fps                             "${FPS}" \
    --border-thickness                "${BORDER_THICKNESS}" \
    --device                          "${DEVICE}" \
    --feature-batch-size              "${FEATURE_BATCH_SIZE}" \
    --action-horizon                  "${ACTION_HORIZON}" \
    --camera-name                     "${CAMERA_NAME}" \
    --transition-proprio-error-weight "${TRANSITION_PROPRIO_ERROR_WEIGHT}" \
    --delta                           "${DELTA}" \
    --delta-step                      "${DELTA_STEP}" \
    --knn-chunk-size                  "${KNN_CHUNK_SIZE}" \
    --lambda-mode                     "${LAMBDA_MODE}" \
    --lambda-window-size              "${LAMBDA_WINDOW_SIZE}" \
    --transition-aux-weight           "${TRANSITION_AUX_WEIGHT}" \
    --calib-fraction                  "${CALIB_FRACTION}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[lpb][viz] wrote to ${OUT_DIR}"
