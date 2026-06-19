#!/usr/bin/env bash
# BCE-WAM benchmark (GT failure split) for robosuite sim data.
#
# Positives  = pre-done frames (is_success==False) from TRAIN success rollouts.
# Eval succ  = same pre-done prefix from a DISJOINT eval success split.
# Failure D_o = GT-failure suffix [first_gt_failure_frame, T) from the fail bank.
# Failure D_e = GT-failure prefix [0, first_gt_failure_frame) joins positives.
#
# Hard constraint: training does NOT run benchmark evaluation. bench.evaluate
# is called by the runner only after fit_on_benchmark returns. Train success and
# eval success come from different splits (no train/val contamination).
#
# Required env:
#   MODEL_CKPT=/path/to/checkpoints/model_<epoch>.pth (optional; hardcoded below)
#     bash robosuite/discriminator/dyn_disc/scripts/run_bce_robosuite_benchmark.sh
#
# Layout assumptions:
#   DATA_ROOT/<task>/fail_rollout-labeled      (BCE failure bank, GT-labeled)
#   DATA_ROOT/<task>/fail_rollout-val-labeled  (benchmark eval failures)
#   DATA_ROOT/<task>/success_rollout           (BCE train positives + calib)
#   DATA_ROOT/<task>/success_rollout-val       (benchmark eval success)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

# ====================================================================== #
# Important parameters: set directly (no ${VAR:-default} indirection).   #
# ====================================================================== #
MODEL_CKPT="checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_10.pth"
TASKS="NutAssemblySquare"
FAIL_SPLIT="fail_rollout-val-labeled"        # eval failures
SUCCESS_SPLIT="success_rollout-val"          # eval success (test only)
SUCCESS_TRAIN_SPLIT="success_rollout"        # BCE train positives + calib
FAIL_TRAIN_SPLIT="fail_rollout-labeled"      # GT-labeled failure bank

# Feature space + head + calibration knobs.
FEATURE_SOURCE="transformer"
TRANSFORMER_LAYER=1
HEAD_HIDDEN=256
HEAD_LAYERS=2
FAIL_BANK_PER_TASK=25
CALIB_MODE="two_class_youden"                # two_class_youden | success_percentile
MAX_EXPERT_OTHER_RATIO=1.0                   # cap |D_e| <= ratio*|D_o|; <=0 disables

# Per-task caps.
TRAIN_MAX_SUCCESS_PER_TASK=50                # train: success_rollout
MAX_FAIL_PER_TASK=50                         # eval: fail_rollout-val-labeled
MAX_SUCCESS_PER_TASK=50                      # eval: success_rollout-val
# ====================================================================== #

if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[robosuite][bce] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

# Output directory.
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)_${TASKS}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/bce_eval_robosuite/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
SAVE_CKPT_DIR="${SAVE_CKPT_DIR:-${OUT_DIR}/checkpoints}"
mkdir -p "${OUT_DIR}"

# Peripheral runtime knobs (still overridable via env).
DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
VISUAL_WEIGHT="${VISUAL_WEIGHT:-1.0}"
PROPRIO_WEIGHT="${PROPRIO_WEIGHT:-1.0}"
ACTION_WEIGHT="${ACTION_WEIGHT:-4.0}"
DELTA="${DELTA:-10.0}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-2048}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"

EXTRA_ARGS=()
if [[ "${TRAIN_MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--train-max-success-per-task "${TRAIN_MAX_SUCCESS_PER_TASK}")
fi
if [[ "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi
if [[ -n "${TASKS}" ]]; then
    EXTRA_ARGS+=(--tasks ${TASKS})
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.robosuite_bce \
    --model-ckpt           "${MODEL_CKPT}" \
    --data-root            "${DATA_ROOT}" \
    --fail-split           "${FAIL_SPLIT}" \
    --success-split        "${SUCCESS_SPLIT}" \
    --success-train-split  "${SUCCESS_TRAIN_SPLIT}" \
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
    --knn-feature-source   "${FEATURE_SOURCE}" \
    --knn-transformer-layer "${TRANSFORMER_LAYER}" \
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
