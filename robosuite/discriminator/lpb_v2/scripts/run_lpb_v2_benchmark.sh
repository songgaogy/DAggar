#!/usr/bin/env bash
# LPB v2 KNN benchmark entry point.
# Run from repo root:
#   bash robosuite/discriminator/lpb_v2/scripts/run_lpb_v2_benchmark.sh
# Override via env vars, e.g.:
#   TASKS="PickPlaceCan" DELTA=5.0 \
#     bash robosuite/discriminator/lpb_v2/scripts/run_lpb_v2_benchmark.sh
#
# Required: MODEL_CKPT must point to a trained dynamics checkpoint .pth.
# Recommended layout (produced by `robosuite.discriminator.lpb_v2.train`):
#   checkpoints/lpb_v2/dynamics/<run_name-timestamp>/{hydra.yaml, normalizer.pth, checkpoints/model_<epoch>.pth}
# Point MODEL_CKPT to:
#   checkpoints/lpb_v2/dynamics/<run_name-timestamp>/checkpoints/model_<epoch>.pth

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/utils/fail_rollout}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/utils/success_rollout}"
SUCCESS_CACHE_ROOT="${SUCCESS_CACHE_ROOT:-${REPO_ROOT}/data/.lpb_score_preprocessed_cache}"
METADATA_CACHE_ROOT="${METADATA_CACHE_ROOT:-${REPO_ROOT}/data/.lpb_score_cache}"
CACHE_CAMERA_NAMES="${CACHE_CAMERA_NAMES:-agentview birdview frontview}"
TASKS="${TASKS:-PickPlaceBread PickPlaceCan PickPlaceCereal PickPlaceMilk}"

MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-100}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-100}"

RUN_NAME="${RUN_NAME:-eval}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/lpb_v2/eval/wam-layers/${RUN_NAME}-${TIMESTAMP}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# You can override this via env var:
#   MODEL_CKPT=/abs/path/checkpoints/lpb_v2/dynamics/<run_name-timestamp>/checkpoints/model_49.pth bash ...
MODEL_CKPT="checkpoints/lpb_v2/dynamics/train-20260428_210501/checkpoints/model_49.pth"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[lpb_v2] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
VISUAL_WEIGHT="${VISUAL_WEIGHT:-1.0}"
PROPRIO_WEIGHT="${PROPRIO_WEIGHT:-1.0}"     # required
ACTION_WEIGHT="${ACTION_WEIGHT:-4.0}"
DELTA="${DELTA:-5.0}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-2048}"
KNN_FEATURE_SOURCE="${KNN_FEATURE_SOURCE:-transformer}"     # transformer / encoder
KNN_TRANSFORMER_LAYER="${KNN_TRANSFORMER_LAYER:-1}"
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
if [[ "${USE_SUCCESS_CACHE:-1}" == "1" && -d "${SUCCESS_CACHE_ROOT}" && -d "${METADATA_CACHE_ROOT}" ]]; then
    EXTRA_ARGS+=(--success-cache-root "${SUCCESS_CACHE_ROOT}")
    EXTRA_ARGS+=(--metadata-cache-root "${METADATA_CACHE_ROOT}")
    if [[ -n "${CACHE_CAMERA_NAMES}" ]]; then
        EXTRA_ARGS+=(--cache-camera-names ${CACHE_CAMERA_NAMES})
    fi
fi

"${PYTHON_BIN}" -m robosuite.discriminator.lpb_v2.run_benchmark \
    --model-ckpt          "${MODEL_CKPT}" \
    --fail-root           "${FAIL_ROOT}" \
    --success-root        "${SUCCESS_ROOT}" \
    --tasks               ${TASKS} \
    --save-json           "${SAVE_JSON}" \
    --device              "${DEVICE}" \
    --encode-batch-size   "${ENCODE_BATCH_SIZE}" \
    --visual-weight       "${VISUAL_WEIGHT}" \
    --proprio-weight      "${PROPRIO_WEIGHT}" \
    --action-weight       "${ACTION_WEIGHT}" \
    --delta               "${DELTA}" \
    --knn-chunk-size      "${KNN_CHUNK_SIZE}" \
    --knn-feature-source  "${KNN_FEATURE_SOURCE}" \
    --knn-transformer-layer "${KNN_TRANSFORMER_LAYER}" \
    --calib-fraction      "${CALIB_FRACTION}" \
    --seed                "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[lpb_v2] wrote ${SAVE_JSON}"
