#!/usr/bin/env bash
# Single-bank KNN benchmark entry point.
# Run from repo root:
#   bash robosuite/discriminator/dyn_disc/scripts/run_single_bank_robosuite_benchmark.sh
#
# Important parameters are hardcoded below (edit them in-place for a run); only
# environment/infrastructure paths (PYTHON_BIN, DATA_ROOT, OUT_DIR) stay
# overridable via env vars. Recommended checkpoint layout (produced by
# `robosuite.discriminator.dyn_disc.training.train`):
#   checkpoints/dyn_disc/dynamics/<run_name-timestamp>/{hydra.yaml, normalizer.pth, checkpoint/model_<epoch>.pth}

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

# --- environment / infrastructure (overridable) ---
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

# --- important parameters (hardcoded) ---
MODEL_CKPT="checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_10.pth"
TASKS="NutAssemblySquare"

FAIL_SPLIT="fail_rollout-val-labeled"        # eval failures
SUCCESS_SPLIT="success_rollout-val"          # eval success (held out from bank/calib)
SUCCESS_TRAIN_SPLIT="success_rollout"        # bank + calibration (disjoint from eval)

MAX_FAIL_PER_TASK=50                         # eval: fail_rollout-val-labeled
MAX_SUCCESS_PER_TASK=50                      # eval: success_rollout-val
TRAIN_MAX_SUCCESS_PER_TASK=50                # train: success_rollout (bank + calib)

DEVICE=cuda
ENCODE_BATCH_SIZE=32
VISUAL_WEIGHT=1.0
PROPRIO_WEIGHT=1.0
ACTION_WEIGHT=1.0
DELTA=5.0
KNN_CHUNK_SIZE=2048
KNN_FEATURE_SOURCE=transformer                # transformer / encoder
KNN_TRANSFORMER_LAYER=1
CALIB_FRACTION=0.2
SEED=0

if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[dyn_disc] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

# --- output directory (pu-bce style, overridable) ---
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-run_${TIMESTAMP}_${TASKS}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/single_bank_eval_robosuite/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
mkdir -p "${OUT_DIR}"

# Optional flags (kept env-overridable; empty by default).
EXTRA_ARGS=()
if [[ "${QUIET_FIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--quiet-fit)
fi
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    EXTRA_ARGS+=(--proprio-indices ${PROPRIO_INDICES})
fi
if [[ -n "${CAMERA_TO_VIEW:-}" ]]; then
    EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.sim_benchmark \
    --model-ckpt          "${MODEL_CKPT}" \
    --data-root           "${DATA_ROOT}" \
    --fail-split          "${FAIL_SPLIT}" \
    --success-split       "${SUCCESS_SPLIT}" \
    --success-train-split "${SUCCESS_TRAIN_SPLIT}" \
    --tasks               ${TASKS} \
    --max-fail-per-task   "${MAX_FAIL_PER_TASK}" \
    --max-success-per-task "${MAX_SUCCESS_PER_TASK}" \
    --train-max-success-per-task "${TRAIN_MAX_SUCCESS_PER_TASK}" \
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

echo "[dyn_disc] wrote ${SAVE_JSON}"
