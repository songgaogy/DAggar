#!/usr/bin/env bash
# Original-LPB KNN benchmark entry point.
# Run from repo root:
#   bash robosuite/discriminator/lpb_original/scripts/run_lpb_original_benchmark.sh
# Override via env vars, e.g.:
#   TASKS="PickPlaceCan" DELTA=5.0 \
#     bash robosuite/discriminator/lpb_original/scripts/run_lpb_original_benchmark.sh
#
# Required: MODEL_CKPT must point to a trained dynamics checkpoint .pth produced
# by lpb_original/train.py (the run dir must also contain hydra.yaml + normalizer.pth).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/utils/fail_rollout}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/utils/success_rollout}"
TASKS="${TASKS:-PickPlaceBread PickPlaceCan PickPlaceCereal PickPlaceMilk}"

MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-100}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-100}"

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/lpb_original/eval/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

MODEL_CKPT="${MODEL_CKPT:?MODEL_CKPT must be set to a .../checkpoints/model_<epoch>.pth produced by lpb_original/train.py}"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[lpb_original] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
VISUAL_WEIGHT="${VISUAL_WEIGHT:-1.0}"
PROPRIO_WEIGHT="${PROPRIO_WEIGHT:-2.0}"
DELTA="${DELTA:-10.0}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-2048}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
SEED="${SEED:-0}"

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
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    EXTRA_ARGS+=(--proprio-indices ${PROPRIO_INDICES})
fi
if [[ -n "${CAMERA_TO_VIEW:-}" ]]; then
    EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
fi

"${PYTHON_BIN}" -m data.utils.benchmark.examples.run_lpb_original \
    --model-ckpt          "${MODEL_CKPT}" \
    --fail-root           "${FAIL_ROOT}" \
    --success-root        "${SUCCESS_ROOT}" \
    --tasks               ${TASKS} \
    --save-json           "${SAVE_JSON}" \
    --device              "${DEVICE}" \
    --encode-batch-size   "${ENCODE_BATCH_SIZE}" \
    --visual-weight       "${VISUAL_WEIGHT}" \
    --proprio-weight      "${PROPRIO_WEIGHT}" \
    --delta               "${DELTA}" \
    --knn-chunk-size      "${KNN_CHUNK_SIZE}" \
    --calib-fraction      "${CALIB_FRACTION}" \
    --seed                "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[lpb_original] wrote ${SAVE_JSON}"
