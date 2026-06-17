#!/usr/bin/env bash
# FLOAT benchmark entry point for the real-world Agilex dataset.
# Run from repo root:
#   bash robosuite/discriminator/float/scripts/run_float_real_world_benchmark.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/agilex/failure_annotations/out_by_task}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/agilex}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/data/.agilex_train_cache}"
TASKS="${TASKS:-candy_in_plate duck_in_bowl Micky_in_box sausage_in_pot}"

MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-25}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-25}"

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/float/real_world_eval/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

ACTION_START="${ACTION_START:-7}"
ACTION_STOP="${ACTION_STOP:-13}"
PROPRIO_FIELD="${PROPRIO_FIELD:-qpos}"
PROPRIO_START="${PROPRIO_START:-7}"
PROPRIO_STOP="${PROPRIO_STOP:-14}"

DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-64}"
CAMERA_NAME="${CAMERA_NAME:-cam_high}"

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
if [[ "${USE_CACHE:-1}" == "1" && -d "${CACHE_ROOT}" ]]; then
    EXTRA_ARGS+=(--cache-root "${CACHE_ROOT}")
fi

"${PYTHON_BIN}" -m benchmark.real_world.examples.run_float \
    --fail-root            "${FAIL_ROOT}" \
    --success-root         "${SUCCESS_ROOT}" \
    --tasks                ${TASKS} \
    --save-json            "${SAVE_JSON}" \
    --proprio-field        "${PROPRIO_FIELD}" \
    --proprio-start        "${PROPRIO_START}" \
    --proprio-stop         "${PROPRIO_STOP}" \
    --action-start         "${ACTION_START}" \
    --action-stop          "${ACTION_STOP}" \
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

echo "[real_world][float] wrote ${SAVE_JSON}"
