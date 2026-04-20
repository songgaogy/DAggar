#!/usr/bin/env bash
# FLOAT (DINOv2) benchmark entry point.
# Run from repo root:
#   bash robosuite/discriminator/float/scripts/run_float_benchmark.sh
# Override via env vars, e.g.:
#   TASKS="PickPlaceCan" STEP_STRIDE=4 MAX_FAIL_PER_TASK=10 MAX_SUCCESS_PER_TASK=15 \
#     bash robosuite/discriminator/float/scripts/run_float_benchmark.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/utils/fail_rollout}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/utils/success_rollout}"
TASKS="${TASKS:-PickPlaceBread PickPlaceCan PickPlaceCereal PickPlaceMilk}"

# Per-task trajectory caps. Empty string or non-positive int = use all trajectories.
MAX_FAIL_PER_TASK="100"
MAX_SUCCESS_PER_TASK="200"

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/float/eval/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-64}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
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

"${PYTHON_BIN}" -m data.utils.benchmark.examples.run_float \
    --fail-root            "${FAIL_ROOT}" \
    --success-root         "${SUCCESS_ROOT}" \
    --tasks                ${TASKS} \
    --save-json            "${SAVE_JSON}" \
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

echo "[float] wrote ${SAVE_JSON}"
