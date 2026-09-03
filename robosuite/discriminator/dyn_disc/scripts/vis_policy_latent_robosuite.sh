#!/usr/bin/env bash
# Visualize frozen policy task_scene_cond features with PCA and t-SNE.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
POLICY_CKPT="${POLICY_CKPT:-${REPO_ROOT}/checkpoints/multitask_6/flow_multi_ep0100.pt}"
TASK="${TASK:-NutAssemblySquare}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/policy/runs/run_${TIMESTAMP}_${TASK}}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/visualizations/latent}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/policy/feature_cache}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
PRELOAD_WORKERS="${PRELOAD_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
REUSE_FEATURE_CACHE="${REUSE_FEATURE_CACHE:-1}"

if [[ ! -f "${POLICY_CKPT}" ]]; then
    echo "[dyn_disc][latent][policy] ERROR: policy checkpoint not found: ${POLICY_CKPT}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}" "${FEATURE_CACHE_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

LOADER_ARGS=(
    --preload-workers "${PRELOAD_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --feature-cache-dir "${FEATURE_CACHE_DIR}"
)
if [[ "${PIN_MEMORY}" == "1" ]]; then
    LOADER_ARGS+=(--pin-memory)
else
    LOADER_ARGS+=(--no-pin-memory)
fi
if [[ "${REUSE_FEATURE_CACHE}" == "1" ]]; then
    LOADER_ARGS+=(--reuse-feature-cache)
else
    LOADER_ARGS+=(--no-reuse-feature-cache)
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.visualization.vis_policy_latent \
    --policy-ckpt "${POLICY_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --fail-split "${FAIL_SPLIT:-fail_rollout-val-labeled}" \
    --success-split "${SUCCESS_SPLIT:-success_rollout-val}" \
    --task "${TASK}" \
    --out-dir "${OUT_DIR}" \
    --max-fail-per-task "${MAX_FAIL_PER_TASK:-50}" \
    --max-success-per-task "${MAX_SUCCESS_PER_TASK:-50}" \
    --device cuda \
    --encode-batch-size "${ENCODE_BATCH_SIZE:-128}" \
    --seed "${SEED:-0}" \
    "${LOADER_ARGS[@]}" \
    "$@"

echo "[dyn_disc][latent][policy] wrote to ${OUT_DIR}"
