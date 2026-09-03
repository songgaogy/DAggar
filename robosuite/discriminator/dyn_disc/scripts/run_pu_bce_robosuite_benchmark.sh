#!/usr/bin/env bash
# nnPU (PU-BCE) benchmark for robosuite sim data.
#
# Positives  = pre-done frames from SUCCESS trajectories (is_success==False).
# Eval succ  = same pre-done prefix; post-done frames are zero-padded (no score).
# Unlabeled  = WHOLE failure-rollout trajectories (NO GT failure timing).
# Calibration = pre-done success frames + per-task success_percentile (no failure labels).
#
# Hard constraint: training does NOT run benchmark evaluation. bench.evaluate
# is called by the runner only after fit_on_benchmark returns.
#
# Required env:
#   MODEL_CKPT=/path/to/checkpoints/model_<epoch>.pth \
#     bash robosuite/discriminator/dyn_disc/scripts/run_pu_bce_robosuite_benchmark.sh
#
# Layout assumptions (override via env vars):
#   DATA_ROOT/<task>/fail_rollout              (UNLABELED failure pool)
#   DATA_ROOT/<task>/fail_rollout-val-labeled  (benchmark eval failures)
#   DATA_ROOT/<task>/success_rollout           (nnPU positives + calibration)
#   DATA_ROOT/<task>/success_rollout-val       (benchmark eval success)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"


MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/ablations/TACO/taco_robosuite-20260903_172611/checkpoint/model_50.pth}"
FAIL_SPLIT="${FAIL_SPLIT:-fail_rollout-val-labeled}"       # benchmark eval failures
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout-val}"      # benchmark eval success
SUCCESS_TRAIN_SPLIT="${SUCCESS_TRAIN_SPLIT:-success_rollout}" # nnPU positives + calibration
FAIL_TRAIN_SPLIT="${FAIL_TRAIN_SPLIT:-fail_rollout}"       # unlabeled failure pool
TASKS="${TASKS:-NutAssemblySquare}"
TRAIN_MAX_SUCCESS_PER_TASK="${TRAIN_MAX_SUCCESS_PER_TASK:-50}" # train success, per task
TRAIN_MAX_FAIL_PER_TASK="${TRAIN_MAX_FAIL_PER_TASK:-50}"       # train failure, per task
MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-50}"                   # eval failure, per task
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-50}"             # eval success, per task

# nnPU + head knobs.
PI_P="${PI_P:-0.3}"         # class prior; set from domain knowledge
# logistic (softplus) surrogate: robust to pi_p and never collapses to the
# trivial risk==pi_p solution. The sigmoid surrogate saturates to zero gradient
# and collapses under low pi_p + weak features (empirically verified).
LOSS_SURROGATE="${LOSS_SURROGATE:-logistic}"
BETA="${BETA:-0.0}"
HEAD_HIDDEN="${HEAD_HIDDEN:-512}"
HEAD_LAYERS="${HEAD_LAYERS:-3}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"


RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)_${TASKS}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/TACO/nnpu/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
SAVE_CKPT_DIR="${SAVE_CKPT_DIR:-${OUT_DIR}/checkpoints}"
mkdir -p "${OUT_DIR}"

if [[ -z "${MODEL_CKPT}" || ! -f "${MODEL_CKPT}" ]]; then
    echo "[robosuite][pu_bce] ERROR: set MODEL_CKPT to a TACO model checkpoint." >&2
    exit 1
fi

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
PRELOAD_WORKERS="${PRELOAD_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
DELTA="${DELTA:-10.0}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
SEED="${SEED:-0}"

EXTRA_ARGS=()
if [[ -n "${TRAIN_MAX_SUCCESS_PER_TASK}" && "${TRAIN_MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--train-max-success-per-task "${TRAIN_MAX_SUCCESS_PER_TASK}")
fi
if [[ -n "${TRAIN_MAX_FAIL_PER_TASK}" && "${TRAIN_MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--train-max-fail-per-task "${TRAIN_MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_FAIL_PER_TASK}" && "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_SUCCESS_PER_TASK}" && "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi
if [[ -n "${TASKS}" ]]; then
    EXTRA_ARGS+=(--tasks ${TASKS})
fi
if [[ "${PIN_MEMORY}" == "1" ]]; then
    EXTRA_ARGS+=(--pin-memory)
else
    EXTRA_ARGS+=(--no-pin-memory)
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.robosuite_pu_bce \
    --model-ckpt            "${MODEL_CKPT}" \
    --data-root             "${DATA_ROOT}" \
    --fail-split            "${FAIL_SPLIT}" \
    --success-split         "${SUCCESS_SPLIT}" \
    --success-train-split   "${SUCCESS_TRAIN_SPLIT}" \
    --fail-train-split      "${FAIL_TRAIN_SPLIT}" \
    --save-json             "${SAVE_JSON}" \
    --save-ckpt-dir         "${SAVE_CKPT_DIR}" \
    --device                "${DEVICE}" \
    --encode-batch-size     "${ENCODE_BATCH_SIZE}" \
    --preload-workers       "${PRELOAD_WORKERS}" \
    --prefetch-factor       "${PREFETCH_FACTOR}" \
    --delta                 "${DELTA}" \
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
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[robosuite][pu_bce] wrote ${SAVE_JSON}"
