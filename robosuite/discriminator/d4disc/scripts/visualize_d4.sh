#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"
PREPROCESSED_CACHE_ROOT="${PREPROCESSED_CACHE_ROOT:-${REPO_ROOT}/data/.lpb_score_preprocessed_cache}"
FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/utils/fail_rollout}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/utils/success_rollout}"

D4_CKPT="${D4_CKPT:?must set D4_CKPT}"
TASK="${TASK:?must set TASK}"

NUM_TRAJS="${NUM_TRAJS:-4}"
OMEGA="${OMEGA:-0.5}"
DELTA="${DELTA:-10.0}"
LAMBDA_MODE="${LAMBDA_MODE:-mean}"
LAMBDA_WINDOW_SIZE="${LAMBDA_WINDOW_SIZE:--1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
FPS="${FPS:-20}"
BORDER_THICKNESS="${BORDER_THICKNESS:-10}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"

SUCC_NUM="${SUCC_NUM:-200}"
FAIL_NUM="${FAIL_NUM:-100}"

DEVICE="${DEVICE:-cuda}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-256}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
SEED="${SEED:-0}"

RUN_NAME="${RUN_NAME:-viz_${TASK}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/d4disc/viz/${RUN_NAME}}"
PDF_NAME="${PDF_NAME:-d4_scores.pdf}"
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${SUCC_NUM}" && "${SUCC_NUM}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${SUCC_NUM}")
fi
if [[ -n "${FAIL_NUM}" && "${FAIL_NUM}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${FAIL_NUM}")
fi
if [[ "${QUIET_FIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--quiet-fit)
fi

"${PYTHON_BIN}" -m robosuite.discriminator.d4disc.visualize \
    --d4-ckpt                "${D4_CKPT}" \
    --preprocessed-cache-root "${PREPROCESSED_CACHE_ROOT}" \
    --fail-root              "${FAIL_ROOT}" \
    --success-root           "${SUCCESS_ROOT}" \
    --task                   "${TASK}" \
    --num-trajs              "${NUM_TRAJS}" \
    --out-dir                "${OUT_DIR}" \
    --pdf-name               "${PDF_NAME}" \
    --seed                   "${SEED}" \
    --fps                    "${FPS}" \
    --border-thickness       "${BORDER_THICKNESS}" \
    --device                 "${DEVICE}" \
    --encoder-batch-size     "${ENCODER_BATCH_SIZE}" \
    --image-size             "${IMAGE_SIZE}" \
    --camera-name            "${CAMERA_NAME}" \
    --omega                  "${OMEGA}" \
    --delta                  "${DELTA}" \
    --lambda-mode            "${LAMBDA_MODE}" \
    --lambda-window-size     "${LAMBDA_WINDOW_SIZE}" \
    --calib-fraction         "${CALIB_FRACTION}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[d4_viz] wrote outputs under ${OUT_DIR}"
