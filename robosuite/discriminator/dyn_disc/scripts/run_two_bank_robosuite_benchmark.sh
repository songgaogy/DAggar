#!/usr/bin/env bash
# Two-bank KNN robosuite benchmark entry point.
# Run from repo root:
#   bash robosuite/discriminator/dyn_disc/scripts/run_two_bank_robosuite_benchmark.sh
# Override via env vars, e.g.:
#   TASKS="PickPlaceCereal" SCORE_MODE=difference ALPHA=1.0 \
#     bash robosuite/discriminator/dyn_disc/scripts/run_two_bank_robosuite_benchmark.sh
#
# Required: MODEL_CKPT must point to a trained dyn_disc dynamics checkpoint .pth.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
FAIL_SPLIT="${FAIL_SPLIT:-fail_rollout-val-labeled}"
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout-val}"
FAIL_TRAIN_SPLIT="${FAIL_TRAIN_SPLIT:-fail_rollout-labeled}"
TASKS="${TASKS:-}"

MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-100}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-100}"

RUN_NAME="${RUN_NAME:-eval_two_bank}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/eval/two-bank/${RUN_NAME}-${TIMESTAMP}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_50.pth}"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[dyn_disc][two_bank] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
VISUAL_WEIGHT="${VISUAL_WEIGHT:-1.0}"
PROPRIO_WEIGHT="${PROPRIO_WEIGHT:-2.0}"
ACTION_WEIGHT="${ACTION_WEIGHT:-1.0}"
DELTA="${DELTA:-10.0}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-2048}"
KNN_FEATURE_SOURCE="${KNN_FEATURE_SOURCE:-transformer}"     # transformer / encoder
KNN_TRANSFORMER_LAYER="${KNN_TRANSFORMER_LAYER:-1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
SEED="${SEED:-0}"

# Two-bank knobs.
FAIL_BANK_PER_TASK="${FAIL_BANK_PER_TASK:-25}"
FAIL_BANK_LAST_K="${FAIL_BANK_LAST_K:-60}"
FAIL_CALIB_PER_TASK="${FAIL_CALIB_PER_TASK:-0}"
SCORE_MODE="${SCORE_MODE:-difference}"          # difference / ratio / dsucc_only
ALPHA="${ALPHA:-1.0}"
CALIB_MODE="${CALIB_MODE:-success_percentile}"  # success_percentile / two_class_youden

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
if [[ -n "${FAIL_BANK_IDS_JSON:-}" ]]; then
    EXTRA_ARGS+=(--fail-bank-ids-json "${FAIL_BANK_IDS_JSON}")
fi
if [[ -n "${TASKS}" ]]; then
    EXTRA_ARGS+=(--tasks ${TASKS})
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.sim_benchmark_two_bank \
    --model-ckpt          "${MODEL_CKPT}" \
    --data-root           "${DATA_ROOT}" \
    --fail-split          "${FAIL_SPLIT}" \
    --success-split       "${SUCCESS_SPLIT}" \
    --fail-train-split    "${FAIL_TRAIN_SPLIT}" \
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
    --fail-bank-per-task  "${FAIL_BANK_PER_TASK}" \
    --fail-bank-last-k    "${FAIL_BANK_LAST_K}" \
    --fail-calib-per-task "${FAIL_CALIB_PER_TASK}" \
    --score-mode          "${SCORE_MODE}" \
    --alpha               "${ALPHA}" \
    --calib-mode          "${CALIB_MODE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[dyn_disc][two_bank] wrote ${SAVE_JSON}"
