#!/usr/bin/env bash
# nnPU (PU-BCE) benchmark for robosuite sim data.
#
# Positives  = frames from SUCCESS trajectories.
# Unlabeled  = WHOLE failure-rollout trajectories (NO GT failure timing).
# Calibration = per-task success_percentile only (no failure labels).
#
# Hard constraint: training does NOT run benchmark evaluation. bench.evaluate
# is called by the runner only after fit_on_benchmark returns.
#
# Required env:
#   MODEL_CKPT=/path/to/checkpoints/model_<epoch>.pth \
#     bash robosuite/discriminator/dyn_disc/scripts/run_pu_bce_robosuite_benchmark.sh
#
# Layout assumptions (override via env vars):
#   DATA_ROOT/<task>/fail_rollout-labeled      (UNLABELED failure pool)
#   DATA_ROOT/<task>/fail_rollout-val-labeled  (benchmark eval failures)
#   DATA_ROOT/<task>/success_rollout-val       (benchmark eval success)

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
FAIL_SPLIT="${FAIL_SPLIT:-fail_rollout-val-labeled}"
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout-val}"
FAIL_TRAIN_SPLIT="${FAIL_TRAIN_SPLIT:-fail_rollout-labeled}"
TASKS="${TASKS:-}"

MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-50}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-50}"

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/pu_bce_eval_robosuite/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
SAVE_CKPT_DIR="${SAVE_CKPT_DIR:-${OUT_DIR}/checkpoints}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_50.pth}"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[robosuite][pu_bce] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
DELTA="${DELTA:-10.0}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-2048}"
KNN_FEATURE_SOURCE="${KNN_FEATURE_SOURCE:-transformer}"
KNN_TRANSFORMER_LAYER="${KNN_TRANSFORMER_LAYER:-1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
SEED="${SEED:-0}"

# nnPU + head knobs.
PI_P="${PI_P:-0.5}"                     # class prior; set from domain knowledge
LOSS_SURROGATE="${LOSS_SURROGATE:-sigmoid}"
BETA="${BETA:-0.0}"
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_LAYERS="${HEAD_LAYERS:-2}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"

# Unlabeled failure pool: drawn from DATA_ROOT/<task>/FAIL_TRAIN_SPLIT.
UNLABELED_PER_TASK="${UNLABELED_PER_TASK:-25}"

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
if [[ "${NO_NN_CORRECTION:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-nn-correction)
fi
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    EXTRA_ARGS+=(--proprio-indices ${PROPRIO_INDICES})
fi
if [[ -n "${CAMERA_TO_VIEW:-}" ]]; then
    EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
fi
if [[ -n "${TASKS}" ]]; then
    EXTRA_ARGS+=(--tasks ${TASKS})
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.robosuite_pu_bce \
    --model-ckpt            "${MODEL_CKPT}" \
    --data-root             "${DATA_ROOT}" \
    --fail-split            "${FAIL_SPLIT}" \
    --success-split         "${SUCCESS_SPLIT}" \
    --fail-train-split      "${FAIL_TRAIN_SPLIT}" \
    --save-json             "${SAVE_JSON}" \
    --save-ckpt-dir         "${SAVE_CKPT_DIR}" \
    --device                "${DEVICE}" \
    --encode-batch-size     "${ENCODE_BATCH_SIZE}" \
    --delta                 "${DELTA}" \
    --knn-chunk-size        "${KNN_CHUNK_SIZE}" \
    --knn-feature-source    "${KNN_FEATURE_SOURCE}" \
    --knn-transformer-layer "${KNN_TRANSFORMER_LAYER}" \
    --calib-fraction        "${CALIB_FRACTION}" \
    --seed                  "${SEED}" \
    --pi-p                  "${PI_P}" \
    --loss-surrogate        "${LOSS_SURROGATE}" \
    --beta                  "${BETA}" \
    --head-hidden           "${HEAD_HIDDEN}" \
    --head-layers           "${HEAD_LAYERS}" \
    --epochs                "${EPOCHS}" \
    --lr                    "${LR}" \
    --weight-decay          "${WEIGHT_DECAY}" \
    --batch-size            "${BATCH_SIZE}" \
    --unlabeled-per-task    "${UNLABELED_PER_TASK}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[robosuite][pu_bce] wrote ${SAVE_JSON}"
