#!/usr/bin/env bash
# Two-bank KNN failure-detector visualization entry point.
# Run from repo root:
#   bash robosuite/discriminator/dyn_disc/scripts/visualize_two_bank_robosuite.sh
#
# Important parameters are hardcoded below; only environment/infrastructure
# paths (PYTHON_BIN, DATA_ROOT, OUT_DIR) stay overridable via env vars.

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

# --- environment / infrastructure (overridable) ---
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

# --- important parameters (hardcoded) ---
MODEL_CKPT="checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_50.pth"
TASK="PickPlaceCereal"

FAIL_SPLIT="fail_rollout-val-labeled"
SUCCESS_SPLIT="success_rollout-val"
FAIL_TRAIN_SPLIT="fail_rollout-labeled"

NUM_TRAJS=3
SEED=0
FPS=20
BORDER_THICKNESS=10
MAX_FAIL_PER_TASK=100
MAX_SUCCESS_PER_TASK=100

DEVICE=cuda
ENCODE_BATCH_SIZE=32
CAMERA_NAME=agentview
VISUAL_WEIGHT=1.0
PROPRIO_WEIGHT=2.0
ACTION_WEIGHT=1.0
DELTA=10.0
KNN_CHUNK_SIZE=2048
KNN_FEATURE_SOURCE=transformer
KNN_TRANSFORMER_LAYER=1
CALIB_FRACTION=0.2
THRESHOLD_SOURCE=detector
STEP_SUCCESS_PERCENTILE=95.0

# Two-bank knobs.
FAIL_BANK_PER_TASK=25
FAIL_BANK_LAST_K=60
FAIL_CALIB_PER_TASK=0
SCORE_MODE=difference
ALPHA=1.0
CALIB_MODE=success_percentile

if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[dyn_disc][viz][two_bank] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

# --- output directory (pu-bce style, overridable) ---
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/two_bank_viz_robosuite/${TASK}-${TIMESTAMP}}"
PDF_NAME="${PDF_NAME:-dyn_disc_two_bank_scores.pdf}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

# Optional flags (kept env-overridable).
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
if [[ "${NO_DEBUG_SCORE_STATS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-debug-score-stats)
fi
if [[ "${THRESHOLD_SOURCE}" == "fixed" ]]; then
    if [[ -z "${FIXED_THRESHOLD:-}" ]]; then
        echo "[dyn_disc][viz][two_bank] ERROR: THRESHOLD_SOURCE=fixed requires FIXED_THRESHOLD." >&2
        exit 1
    fi
    EXTRA_ARGS+=(--fixed-threshold "${FIXED_THRESHOLD}")
fi
if [[ -n "${BENCHMARK_JSON:-}" ]]; then
    EXTRA_ARGS+=(--benchmark-json "${BENCHMARK_JSON}")
fi
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    EXTRA_ARGS+=(--proprio-indices ${PROPRIO_INDICES})
fi
if [[ -n "${CAMERA_TO_VIEW:-}" ]]; then
    EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.visualization.visualize \
    --mode               two_bank \
    --model-ckpt         "${MODEL_CKPT}" \
    --data-root          "${DATA_ROOT}" \
    --fail-split         "${FAIL_SPLIT}" \
    --success-split      "${SUCCESS_SPLIT}" \
    --fail-train-split   "${FAIL_TRAIN_SPLIT}" \
    --task               "${TASK}" \
    --num-trajs          "${NUM_TRAJS}" \
    --out-dir            "${OUT_DIR}" \
    --pdf-name           "${PDF_NAME}" \
    --seed               "${SEED}" \
    --fps                "${FPS}" \
    --border-thickness   "${BORDER_THICKNESS}" \
    --device             "${DEVICE}" \
    --encode-batch-size  "${ENCODE_BATCH_SIZE}" \
    --camera-name        "${CAMERA_NAME}" \
    --visual-weight      "${VISUAL_WEIGHT}" \
    --proprio-weight     "${PROPRIO_WEIGHT}" \
    --action-weight      "${ACTION_WEIGHT}" \
    --delta              "${DELTA}" \
    --knn-chunk-size     "${KNN_CHUNK_SIZE}" \
    --knn-feature-source "${KNN_FEATURE_SOURCE}" \
    --knn-transformer-layer "${KNN_TRANSFORMER_LAYER}" \
    --calib-fraction     "${CALIB_FRACTION}" \
    --threshold-source   "${THRESHOLD_SOURCE}" \
    --step-success-percentile "${STEP_SUCCESS_PERCENTILE}" \
    --fail-bank-per-task "${FAIL_BANK_PER_TASK}" \
    --fail-bank-last-k   "${FAIL_BANK_LAST_K}" \
    --fail-calib-per-task "${FAIL_CALIB_PER_TASK}" \
    --score-mode         "${SCORE_MODE}" \
    --alpha              "${ALPHA}" \
    --calib-mode         "${CALIB_MODE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[dyn_disc][viz][two_bank] wrote to ${OUT_DIR}"
