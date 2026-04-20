#!/usr/bin/env bash
# logpZO failure-detector visualization entry point.
# Run from repo root:
#   bash robosuite/discriminator/logpZO/scripts/visualize_logpZO.sh
# Overrides via env vars, e.g.:
#   TASK=PickPlaceCan NUM_TRAJS=6 SEED=42 EPOCHS=30 MAX_SUCCESS_PER_TASK=30 \
#     bash robosuite/discriminator/logpZO/scripts/visualize_logpZO.sh

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
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/logpZO/viz/${RUN_NAME}}"
PDF_NAME="${PDF_NAME:-logpZO_scores.pdf}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# Encoder.
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-64}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"

# Flow / training.
NUM_LAYERS="${NUM_LAYERS:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
SCALE_CLAMP="${SCALE_CLAMP:-3.0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
ALPHA="${ALPHA:-0.1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"

EXTRA_ARGS=()
if [[ -n "${MAX_FAIL_PER_TASK}" && "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_SUCCESS_PER_TASK}" && "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi
if [[ "${QUIET_FIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--quiet-fit)
fi

"${PYTHON_BIN}" -m robosuite.discriminator.logpZO.visualize \
    --fail-root            "${FAIL_ROOT}" \
    --success-root         "${SUCCESS_ROOT}" \
    --task                 "${TASK}" \
    --num-trajs            "${NUM_TRAJS}" \
    --out-dir              "${OUT_DIR}" \
    --pdf-name             "${PDF_NAME}" \
    --seed                 "${SEED}" \
    --fps                  "${FPS}" \
    --border-thickness     "${BORDER_THICKNESS}" \
    --device               "${DEVICE}" \
    --image-size           "${IMAGE_SIZE}" \
    --encoder-batch-size   "${ENCODER_BATCH_SIZE}" \
    --camera-name          "${CAMERA_NAME}" \
    --num-layers           "${NUM_LAYERS}" \
    --hidden-dim           "${HIDDEN_DIM}" \
    --scale-clamp          "${SCALE_CLAMP}" \
    --epochs               "${EPOCHS}" \
    --batch-size           "${BATCH_SIZE}" \
    --lr                   "${LR}" \
    --weight-decay         "${WEIGHT_DECAY}" \
    --val-fraction         "${VAL_FRACTION}" \
    --early-stop-patience  "${EARLY_STOP_PATIENCE}" \
    --alpha                "${ALPHA}" \
    --calib-fraction       "${CALIB_FRACTION}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[logpZO][viz] wrote to ${OUT_DIR}"
