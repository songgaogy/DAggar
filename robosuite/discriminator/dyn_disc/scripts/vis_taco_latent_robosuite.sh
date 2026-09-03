#!/usr/bin/env bash
# Visualize TACO encoder latents with PCA and t-SNE (CUDA inference only).
# MODEL_CKPT may be overridden to select a particular pretraining checkpoint.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MODEL_CKPT="${MODEL_CKPT:-}"
TASK="${TASK:-NutAssemblyRound}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/TACO/visualizations/latent/${TASK}-${TIMESTAMP}}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

if [[ -z "${MODEL_CKPT}" || ! -f "${MODEL_CKPT}" ]]; then
    echo "[dyn_disc][latent][taco] ERROR: set MODEL_CKPT to a TACO model checkpoint." >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.visualization.vis_taco_latent \
    --model-ckpt "${MODEL_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --fail-split "${FAIL_SPLIT:-fail_rollout-val-labeled}" \
    --success-split "${SUCCESS_SPLIT:-success_rollout-val}" \
    --task "${TASK}" \
    --out-dir "${OUT_DIR}" \
    --max-fail-per-task "${MAX_FAIL_PER_TASK:-50}" \
    --max-success-per-task "${MAX_SUCCESS_PER_TASK:-50}" \
    --device cuda \
    --encode-batch-size "${ENCODE_BATCH_SIZE:-32}" \
    --seed "${SEED:-0}" \
    "$@"

echo "[dyn_disc][latent][taco] wrote to ${OUT_DIR}"
