#!/usr/bin/env bash
# LPB v2 KNN failure-detector visualization entry point.
# Run from repo root:
#   bash robosuite/discriminator/dyn_disc/scripts/visualize_dyn_disc.sh
# Override via env vars, e.g.:
#   TASK=PickPlaceCan NUM_TRAJS=3 DELTA=5.0 \
#     bash robosuite/discriminator/dyn_disc/scripts/visualize_dyn_disc.sh
#
# Required: MODEL_CKPT must point to a trained LPB v2 dynamics checkpoint .pth.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
FAIL_SPLIT="${FAIL_SPLIT:-fail_rollout-val-labeled}"
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout-val}"

TASK="${TASK:-PickPlaceCereal}"
NUM_TRAJS="${NUM_TRAJS:-3}"
SEED="${SEED:-0}"
FPS="${FPS:-20}"
BORDER_THICKNESS="${BORDER_THICKNESS:-10}"

MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-100}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-100}"

RUN_NAME="${RUN_NAME:-viz_${TASK}}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/viz/${RUN_NAME}-${TIMESTAMP}}"
PDF_NAME="${PDF_NAME:-dyn_disc_scores.pdf}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/dynamics/train-20260427_014122/checkpoints/model_49.pth}"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[dyn_disc][viz] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    echo "               Set MODEL_CKPT=/abs/path/to/checkpoints/model_<epoch>.pth." >&2
    exit 1
fi

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
VISUAL_WEIGHT="${VISUAL_WEIGHT:-1.0}"
PROPRIO_WEIGHT="${PROPRIO_WEIGHT:-2.0}"
DELTA="${DELTA:-10.0}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-2048}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
THRESHOLD_SOURCE="${THRESHOLD_SOURCE:-detector}"
STEP_SUCCESS_PERCENTILE="${STEP_SUCCESS_PERCENTILE:-95.0}"

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
        echo "[dyn_disc][viz] ERROR: THRESHOLD_SOURCE=fixed requires FIXED_THRESHOLD." >&2
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
    --model-ckpt         "${MODEL_CKPT}" \
    --data-root          "${DATA_ROOT}" \
    --fail-split         "${FAIL_SPLIT}" \
    --success-split      "${SUCCESS_SPLIT}" \
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
    --delta              "${DELTA}" \
    --knn-chunk-size     "${KNN_CHUNK_SIZE}" \
    --calib-fraction     "${CALIB_FRACTION}" \
    --threshold-source   "${THRESHOLD_SOURCE}" \
    --step-success-percentile "${STEP_SUCCESS_PERCENTILE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[dyn_disc][viz] wrote to ${OUT_DIR}"
