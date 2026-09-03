#!/usr/bin/env bash
# Visualize PU-BCE scores on robosuite trajectories (MP4 + PDF).
#
# Required env:
#   LOAD_CKPT=/path/to/pu_bce_head.pth \
#   MODEL_CKPT=/path/to/model_<epoch>.pth \
#     bash robosuite/discriminator/dyn_disc/scripts/visualize_pu_bce_robosuite.sh
#
# Optional: TASK, SPLIT (fail_rollout|success_rollout|both), NUM_TRAJS (per split when both),
#   OUT_DIR, CAMERA_NAME, MAX_FAIL_PER_TASK, MAX_SUCCESS_PER_TASK, DATA_ROOT

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"


LOAD_CKPT="${LOAD_CKPT:-}"
MODEL_CKPT="${MODEL_CKPT:-}"
TASK="${TASK:-NutAssemblySquare}"
NUM_TRAJS="${NUM_TRAJS:-5}"     # per split for each
FAIL_SPLIT="${FAIL_SPLIT:-fail_rollout-val-labeled}"
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout-val}"


TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/TACO/visualizations/video/${TASK}-${TIMESTAMP}}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

if [[ -z "${MODEL_CKPT}" || ! -f "${MODEL_CKPT}" ]]; then
    echo "[pu_bce][viz] ERROR: set MODEL_CKPT to a TACO model checkpoint." >&2
    exit 1
fi
if [[ -z "${LOAD_CKPT}" || ! -f "${LOAD_CKPT}" ]]; then
    echo "[pu_bce][viz] ERROR: set LOAD_CKPT to the matching nnPU head checkpoint." >&2
    exit 1
fi

EXTRA_ARGS=(--quiet-fit)
if [[ -n "${MAX_FAIL_PER_TASK:-}" && "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_SUCCESS_PER_TASK:-}" && "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi


"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.visualization.visualize_pu_bce \
    --model-ckpt        "${MODEL_CKPT}" \
    --load-ckpt         "${LOAD_CKPT}" \
    --data-root         "${DATA_ROOT}" \
    --fail-split        "${FAIL_SPLIT}" \
    --success-split     "${SUCCESS_SPLIT}" \
    --task              "${TASK}" \
    --num-trajs         "${NUM_TRAJS}" \
    --out-dir           "${OUT_DIR}" \
    --device            "${DEVICE:-cuda}" \
    --encode-batch-size "${ENCODE_BATCH_SIZE:-32}" \
    --camera-name       "${CAMERA_NAME:-agentview}" \
    --seed              "${SEED:-0}" \
    --fps               "${FPS:-20}" \
    --border-thickness  "${BORDER_THICKNESS:-10}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[pu_bce][viz] wrote to ${OUT_DIR}"
