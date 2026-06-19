#!/usr/bin/env bash
# Two-bank KNN robosuite benchmark entry point.
# Run from repo root:
#   bash robosuite/discriminator/dyn_disc/scripts/run_two_bank_robosuite_benchmark.sh
#
# Important parameters are hardcoded below (edit them in-place for a run); only
# environment/infrastructure paths (PYTHON_BIN, DATA_ROOT, OUT_DIR) stay
# overridable via env vars.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

# --- environment / infrastructure (overridable) ---
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

# --- important parameters (hardcoded) ---
MODEL_CKPT="checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_50.pth"
TASKS="PickPlaceCereal"

FAIL_SPLIT="fail_rollout-val-labeled"        # eval failures
SUCCESS_SPLIT="success_rollout-val"          # eval success (held out from bank/calib)
SUCCESS_TRAIN_SPLIT="success_rollout"        # success bank + calibration (disjoint from eval)
FAIL_TRAIN_SPLIT="fail_rollout-labeled"      # GT-labeled failure bank source

MAX_FAIL_PER_TASK=100                         # eval: fail_rollout-val-labeled
MAX_SUCCESS_PER_TASK=100                      # eval: success_rollout-val
TRAIN_MAX_SUCCESS_PER_TASK=100                # train: success_rollout (bank + calib)

DEVICE=cuda
ENCODE_BATCH_SIZE=32
VISUAL_WEIGHT=1.0
PROPRIO_WEIGHT=2.0
ACTION_WEIGHT=1.0
DELTA=10.0
KNN_CHUNK_SIZE=2048
KNN_FEATURE_SOURCE=transformer                # transformer / encoder
KNN_TRANSFORMER_LAYER=1
CALIB_FRACTION=0.2
SEED=0

# Two-bank knobs.
FAIL_BANK_PER_TASK=25
FAIL_BANK_LAST_K=60
FAIL_CALIB_PER_TASK=0
SCORE_MODE=difference                         # difference / ratio / dsucc_only
ALPHA=1.0
CALIB_MODE=success_percentile                 # success_percentile / two_class_youden

if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[dyn_disc][two_bank] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

# --- output directory (pu-bce style, overridable) ---
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-run_${TIMESTAMP}_${TASKS}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/two_bank_eval_robosuite/${RUN_NAME}}"
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
if [[ -n "${FAIL_BANK_IDS_JSON:-}" ]]; then
    EXTRA_ARGS+=(--fail-bank-ids-json "${FAIL_BANK_IDS_JSON}")
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.sim_benchmark_two_bank \
    --model-ckpt          "${MODEL_CKPT}" \
    --data-root           "${DATA_ROOT}" \
    --fail-split          "${FAIL_SPLIT}" \
    --success-split       "${SUCCESS_SPLIT}" \
    --success-train-split "${SUCCESS_TRAIN_SPLIT}" \
    --fail-train-split    "${FAIL_TRAIN_SPLIT}" \
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
    --fail-bank-per-task  "${FAIL_BANK_PER_TASK}" \
    --fail-bank-last-k    "${FAIL_BANK_LAST_K}" \
    --fail-calib-per-task "${FAIL_CALIB_PER_TASK}" \
    --score-mode          "${SCORE_MODE}" \
    --alpha               "${ALPHA}" \
    --calib-mode          "${CALIB_MODE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[dyn_disc][two_bank] wrote ${SAVE_JSON}"
