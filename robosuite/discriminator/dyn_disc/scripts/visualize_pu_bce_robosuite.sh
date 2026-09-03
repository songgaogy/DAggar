#!/usr/bin/env bash
# Visualize PU-BCE scores on robosuite trajectories (MP4 + PDF).
#
# Required env:
#   LOAD_CKPT=/path/to/pu_bce_head.pth \
#     bash robosuite/discriminator/dyn_disc/scripts/visualize_pu_bce_robosuite.sh
#
# Optional: POLICY_CKPT, TASK, SPLIT (fail_rollout|success_rollout|both),
#   NUM_TRAJS (per split when both), OUT_DIR, CAMERA_NAME,
#   MAX_FAIL_PER_TASK, MAX_SUCCESS_PER_TASK, DATA_ROOT

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"


LOAD_CKPT="${LOAD_CKPT:?Set LOAD_CKPT to a trained policy-feature pu_bce_head.pth}"
POLICY_CKPT="${POLICY_CKPT:-checkpoints/multitask_6/flow_multi_ep0100.pt}"
TASK="${TASK:-NutAssemblySquare}"
NUM_TRAJS=5     # per split for each
FAIL_SPLIT="fail_rollout-val-labeled"
SUCCESS_SPLIT="success_rollout-val"


TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-run_${TIMESTAMP}_${TASK}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/policy/runs/${RUN_NAME}/visualizations}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

EXTRA_ARGS=(--quiet-fit)
if [[ "${PIN_MEMORY:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--pin-memory)
else
    EXTRA_ARGS+=(--no-pin-memory)
fi
if [[ "${REUSE_FEATURE_CACHE:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--reuse-feature-cache)
else
    EXTRA_ARGS+=(--no-reuse-feature-cache)
fi
if [[ -n "${MAX_FAIL_PER_TASK:-}" && "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_SUCCESS_PER_TASK:-}" && "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi


"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.visualization.visualize_pu_bce \
    --policy-ckpt       "${POLICY_CKPT}" \
    --load-ckpt         "${LOAD_CKPT}" \
    --data-root         "${DATA_ROOT}" \
    --fail-split        "${FAIL_SPLIT}" \
    --success-split     "${SUCCESS_SPLIT}" \
    --task              "${TASK}" \
    --num-trajs         "${NUM_TRAJS}" \
    --out-dir           "${OUT_DIR}" \
    --device            "${DEVICE:-cuda}" \
    --encode-batch-size "${ENCODE_BATCH_SIZE:-128}" \
    --preload-workers   "${PRELOAD_WORKERS:-4}" \
    --prefetch-factor   "${PREFETCH_FACTOR:-2}" \
    --feature-cache-dir "${FEATURE_CACHE_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/policy/feature_cache}" \
    --camera-name       "${CAMERA_NAME:-agentview}" \
    --seed              "${SEED:-0}" \
    --fps               "${FPS:-20}" \
    --border-thickness  "${BORDER_THICKNESS:-10}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[pu_bce][viz] wrote to ${OUT_DIR}"
