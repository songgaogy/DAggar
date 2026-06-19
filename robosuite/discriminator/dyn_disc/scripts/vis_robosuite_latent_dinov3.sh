#!/usr/bin/env bash
# DINOv3 WAM latent visualization for robosuite sim benchmark data.
# Run from repo root:
#   bash robosuite/discriminator/dyn_disc/scripts/vis_robosuite_latent_dinov3.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=1


RUN_DIR="checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518"
EPOCH=10
TASK="PickPlaceCereal"
MAX_FAIL_PER_TASK=50
MAX_SUCCESS_PER_TASK=50
ENCODE_BATCH_SIZE=32
FEATURE_SOURCE="encoder"
LAYER=0


MODEL_CKPT="${RUN_DIR}/checkpoint/model_${EPOCH}.pth"
OUT_DIR="${RUN_DIR}/vis-latent/${FEATURE_SOURCE}_layer${LAYER}_${TASK}_ep${EPOCH}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-dinov3-robosuite-latent}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[dyn_disc][latent][dinov3][robosuite] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.visualization.vis_robosuite_latent_dinov3 \
    --model-ckpt            "${MODEL_CKPT}" \
    --data-root             "${DATA_ROOT}" \
    --task                 ${TASK} \
    --out-dir               "${OUT_DIR}" \
    --max-fail-per-task     "${MAX_FAIL_PER_TASK}" \
    --max-success-per-task  "${MAX_SUCCESS_PER_TASK}" \
    --device                "${DEVICE}" \
    --encode-batch-size     "${ENCODE_BATCH_SIZE}" \
    --knn-feature-source    "${FEATURE_SOURCE}" \
    --knn-transformer-layer "${LAYER}" \
    --seed                  "${SEED}" \
    "$@"

echo "[dyn_disc][latent][dinov3][robosuite] wrote to ${OUT_DIR}"
