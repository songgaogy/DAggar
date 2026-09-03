#!/usr/bin/env bash
# Generate PCA/t-SNE plots and an NPZ export of RPT action-token features.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=1

MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/ablations/RPT/pretrain/rpt_robosuite-20260903_192142/checkpoint/model_50.pth}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
TASK="${TASK:-PickPlaceCereal}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/RPT/visualizations/latent/${TASK}-${TIMESTAMP}}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"

"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
[[ -f "${MODEL_CKPT}" ]] || { echo "RPT checkpoint not found: ${MODEL_CKPT}" >&2; exit 1; }

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.visualization.visualize_rpt_latents \
    --model-ckpt "${MODEL_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --task "${TASK}" \
    --fail-split "${FAIL_SPLIT:-fail_rollout-val-labeled}" \
    --success-split "${SUCCESS_SPLIT:-success_rollout-val}" \
    --num-trajs "${NUM_TRAJS:-8}" \
    --max-points "${MAX_POINTS:-10000}" \
    --out-dir "${OUT_DIR}" \
    --device cuda \
    --encode-batch-size "${ENCODE_BATCH_SIZE:-128}" \
    --seed "${SEED:-0}" \
    "$@"

echo "[rpt][latent-viz] wrote ${OUT_DIR}"
