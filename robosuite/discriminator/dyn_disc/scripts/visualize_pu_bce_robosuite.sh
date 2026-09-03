#!/usr/bin/env bash
# Render nnPU score videos and a PDF using an RPT representation checkpoint.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
MODEL_CKPT="${MODEL_CKPT:?Set MODEL_CKPT to an RPT model checkpoint}"
LOAD_CKPT="${LOAD_CKPT:?Set LOAD_CKPT to a fitted pu_bce_head.pth}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
TASK="${TASK:-NutAssemblySquare}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/RPT/visualizations/video/${TASK}-${TIMESTAMP}}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"

"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
[[ -f "${MODEL_CKPT}" ]] || { echo "RPT checkpoint not found: ${MODEL_CKPT}" >&2; exit 1; }
[[ -f "${LOAD_CKPT}" ]] || { echo "nnPU checkpoint not found: ${LOAD_CKPT}" >&2; exit 1; }

EXTRA_ARGS=(--quiet-fit)
if [[ "${NO_FLIP_VERTICAL:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-flip-vertical)
fi
if [[ -n "${MAX_FAIL_PER_TASK:-}" && "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_SUCCESS_PER_TASK:-}" && "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.visualization.visualize_pu_bce \
    --model-ckpt "${MODEL_CKPT}" \
    --load-ckpt "${LOAD_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --fail-split "${FAIL_SPLIT:-fail_rollout-val-labeled}" \
    --success-split "${SUCCESS_SPLIT:-success_rollout-val}" \
    --task "${TASK}" \
    --split "${SPLIT:-both}" \
    --num-trajs "${NUM_TRAJS:-5}" \
    --out-dir "${OUT_DIR}" \
    --device cuda \
    --encode-batch-size "${ENCODE_BATCH_SIZE:-128}" \
    --camera-name "${CAMERA_NAME:-agentview}" \
    --seed "${SEED:-0}" \
    --fps "${FPS:-20}" \
    --border-thickness "${BORDER_THICKNESS:-10}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[rpt][video-viz] wrote ${OUT_DIR}"
