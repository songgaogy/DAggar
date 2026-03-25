#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

SEED=42
CKPT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt"
LPB_CKPT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/lpb_new/lpb_new_20260324_232921/lpb_new_20260324_232921_ep0050.pt"
SAVE_DIR="${SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_new/visualize}"
IMAGE_SIZE=128
ENCODER_BATCH_SIZE=96
NUM_VIDEOS=10
FPS=20
CAMERA_NAME="agentview"

if [[ -z "${LPB_CKPT}" ]]; then
  LPB_CKPT="$(find "${ROOT}/checkpoints/multitask_6/lpb_new" -maxdepth 2 -type f -name 'lpb_new_*.pt' ! -name '*_ep*.pt' | sort | tail -n 1)"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[visualize_lpb_new] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

if [[ -z "${LPB_CKPT}" || ! -f "${LPB_CKPT}" ]]; then
  echo "[visualize_lpb_new] Missing LPB checkpoint. Set LPB_CKPT=/abs/path/to/lpb_new_XXXX.pt" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0

echo "[visualize_lpb_new] ROOT=${ROOT}"
echo "[visualize_lpb_new] policy.ckpt=${CKPT}"
echo "[visualize_lpb_new] model.lpb_ckpt=${LPB_CKPT}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_new/visualize_failures.py" \
  seed="${SEED}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${CKPT}" \
  policy.encoder_batch_size="${ENCODER_BATCH_SIZE}" \
  data.image_size="${IMAGE_SIZE}" \
  model.lpb_ckpt="${LPB_CKPT}" \
  visualization.num_videos="${NUM_VIDEOS}" \
  visualization.fps="${FPS}" \
  visualization.camera_name="${CAMERA_NAME}" \
  "$@"
