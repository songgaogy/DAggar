#!/usr/bin/env bash
# BCE-WAM benchmark (GT failure split) for robosuite sim data.
# Hard constraint: training does NOT run benchmark evaluation. bench.evaluate
# is called by the runner only after fit_on_benchmark returns.
#
# Required env:
#   MODEL_CKPT=/path/to/checkpoints/model_<epoch>.pth \
#     bash robosuite/discriminator/dyn_disc/scripts/run_bce_robosuite_benchmark.sh
#
# Layout assumptions (override via env vars):
#   DATA_ROOT/<task>/fail_rollout-labeled      (BCE failure bank)
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

# --------------------------------------------------------
# for evaluation
MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-50}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-50}"
# --------------------------------------------------------

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/bce_eval_robosuite/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
SAVE_CKPT_DIR="${SAVE_CKPT_DIR:-${OUT_DIR}/checkpoints}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/dynamics/train-20260428_210501/checkpoints/model_49.pth}"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[robosuite][bce] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
VISUAL_WEIGHT="${VISUAL_WEIGHT:-1.0}"
PROPRIO_WEIGHT="${PROPRIO_WEIGHT:-1.0}"
ACTION_WEIGHT="${ACTION_WEIGHT:-4.0}"
DELTA="${DELTA:-10.0}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-2048}"
# Locks the feature space to transformer layer 1 (matches diagnostic + two-bank).
KNN_FEATURE_SOURCE="${KNN_FEATURE_SOURCE:-transformer}"
KNN_TRANSFORMER_LAYER="${KNN_TRANSFORMER_LAYER:-1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
SEED="${SEED:-0}"

# --------------------------------------------------------
# BCE head + optim knobs.
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_LAYERS="${HEAD_LAYERS:-2}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"

# Balance: MAX_EXPERT_OTHER_RATIO caps |D_e| <= ratio * |D_o|. <=0 disables.
MAX_EXPERT_OTHER_RATIO="${MAX_EXPERT_OTHER_RATIO:-1.0}"

# Failure pool: drawn from DATA_ROOT/<task>/FAIL_TRAIN_SPLIT.
# If a task has fewer GT-labeled failures than FAIL_BANK_PER_TASK, all are used.
FAIL_BANK_PER_TASK="${FAIL_BANK_PER_TASK:-25}"

# Per-task threshold calibration. AUROC / AUPRC are computed from continuous
# step_scores, so they are invariant under CALIB_MODE; only F1 / precision /
# recall move with tau.
#   success_percentile : tau = percentile(success-calib failure scores, 100 - DELTA)
#   two_class_youden   : tau = argmax(TPR - FPR) on (success-calib, fail-suffix)
CALIB_MODE="${CALIB_MODE:-two_class_youden}"
# --------------------------------------------------------


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
if [[ -n "${TASKS}" ]]; then
    EXTRA_ARGS+=(--tasks ${TASKS})
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.robosuite_bce \
    --model-ckpt           "${MODEL_CKPT}" \
    --data-root            "${DATA_ROOT}" \
    --fail-split           "${FAIL_SPLIT}" \
    --success-split        "${SUCCESS_SPLIT}" \
    --fail-train-split     "${FAIL_TRAIN_SPLIT}" \
    --save-json            "${SAVE_JSON}" \
    --save-ckpt-dir        "${SAVE_CKPT_DIR}" \
    --device               "${DEVICE}" \
    --encode-batch-size    "${ENCODE_BATCH_SIZE}" \
    --visual-weight        "${VISUAL_WEIGHT}" \
    --proprio-weight       "${PROPRIO_WEIGHT}" \
    --action-weight        "${ACTION_WEIGHT}" \
    --delta                "${DELTA}" \
    --knn-chunk-size       "${KNN_CHUNK_SIZE}" \
    --knn-feature-source   "${KNN_FEATURE_SOURCE}" \
    --knn-transformer-layer "${KNN_TRANSFORMER_LAYER}" \
    --calib-fraction       "${CALIB_FRACTION}" \
    --seed                 "${SEED}" \
    --head-hidden          "${HEAD_HIDDEN}" \
    --head-layers          "${HEAD_LAYERS}" \
    --epochs               "${EPOCHS}" \
    --lr                   "${LR}" \
    --weight-decay         "${WEIGHT_DECAY}" \
    --batch-size           "${BATCH_SIZE}" \
    --max-expert-other-ratio "${MAX_EXPERT_OTHER_RATIO}" \
    --fail-bank-per-task   "${FAIL_BANK_PER_TASK}" \
    --calib-mode           "${CALIB_MODE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[robosuite][bce] wrote ${SAVE_JSON}"
