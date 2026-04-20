#!/usr/bin/env bash
# FLOAT failure-detector visualization entry point.
# Run from repo root:
#   bash robosuite/discriminator/float/scripts/visualize_float.sh
# Overrides via env vars, e.g.:
#   TASK=PickPlaceCan NUM_TRAJS=6 SEED=42 STEP_STRIDE=4 MAX_SUCCESS_PER_TASK=30 \
#     bash robosuite/discriminator/float/scripts/visualize_float.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/utils/fail_rollout}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/utils/success_rollout}"
TASK="${TASK:-PickPlaceMilk}"
NUM_TRAJS="${NUM_TRAJS:-3}"
SEED="${SEED:-0}"
FPS="${FPS:-20}"
BORDER_THICKNESS="${BORDER_THICKNESS:-10}"

# Per-task caps (control fit/scoring scale). Empty or <=0 = use all.
MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-100}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-200}"

RUN_NAME="${RUN_NAME:-viz_${TASK}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/float/viz/${RUN_NAME}}"
PDF_NAME="${PDF_NAME:-float_scores.pdf}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# Encoder.
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-64}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"

# FLOAT OT knobs.
SINKHORN_REG="${SINKHORN_REG:-0.05}"
MAX_ITER="${MAX_ITER:-300}"
TOL="${TOL:-1e-5}"
DELTA="${DELTA:-10.0}"
STEP_STRIDE="${STEP_STRIDE:-8}"

EXTRA_ARGS=()
if [[ -n "${MAX_FAIL_PER_TASK}" && "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_SUCCESS_PER_TASK}" && "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi
if [[ "${USE_SIMILARITY_COST:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--use-similarity-cost)
fi
if [[ "${NO_FLIP_VERTICAL:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-flip-vertical)
fi

"${PYTHON_BIN}" -m robosuite.discriminator.float.visualize \
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
    --sinkhorn-reg         "${SINKHORN_REG}" \
    --max-iter             "${MAX_ITER}" \
    --tol                  "${TOL}" \
    --delta                "${DELTA}" \
    --step-stride          "${STEP_STRIDE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[float][viz] wrote to ${OUT_DIR}"
