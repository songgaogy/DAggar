#!/usr/bin/env bash
# Visualize an existing PU-BCE checkpoint on robosuite trajectories (MP4 + PDF).

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
LOAD_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite-chunk_v2-P98/run_20260719_204315_PickPlaceCereal/checkpoints/pu_bce_head.pth"
MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_10.pth}"
TASK="${TASK:-PickPlaceCereal}"
SPLIT="${SPLIT:-both}"
NUM_TRAJS="${NUM_TRAJS:-10}"
FAIL_SPLIT="${FAIL_SPLIT:-fail_rollout-val-labeled}"
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout-val}"

# Default: write next to the loaded run checkpoint as run_*/vis/
LOAD_CKPT_ABS="$(cd "$(dirname "${LOAD_CKPT}")" && pwd)/$(basename "${LOAD_CKPT}")"
RUN_DIR="$(cd "$(dirname "${LOAD_CKPT_ABS}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/vis}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

EXTRA_ARGS=()
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
    --split             "${SPLIT}" \
    --num-trajs         "${NUM_TRAJS}" \
    --out-dir           "${OUT_DIR}" \
    --device            "${DEVICE:-cuda}" \
    --encode-batch-size "${ENCODE_BATCH_SIZE:-32}" \
    --camera-name       "${CAMERA_NAME:-agentview}" \
    --feature-source    "${FEATURE_SOURCE:-transformer}" \
    --transformer-layer "${TRANSFORMER_LAYER:-1}" \
    --seed              "${SEED:-0}" \
    --fps               "${FPS:-20}" \
    --border-thickness  "${BORDER_THICKNESS:-10}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[pu_bce][viz] wrote to ${OUT_DIR}"
