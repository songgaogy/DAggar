#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

GPU=1
SEED=1
POLICY_CKPT="${POLICY_CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
DICE_CKPT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/lpb_dice/lpb_dice_20260410_132822/lpb_dice_20260410_132822.pt"
SAVE_DIR="${SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_dice/visualize}"
NUM_VIDEOS=12
SUPPORT_PENALTY_WEIGHT=0  # transition error


IMAGE_SIZE="${IMAGE_SIZE:-128}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-96}"
FPS="${FPS:-20}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
SAVE_PDF="${SAVE_PDF:-true}"
NUM_PLOT_FRAMES="${NUM_PLOT_FRAMES:-8}"
DATA_SOURCE="fail_rollout"
SOURCE_SPLIT="${SOURCE_SPLIT:-all}"

if [[ ! -f "${POLICY_CKPT}" ]]; then
  echo "[visualize_lpb_dice] Missing policy checkpoint: ${POLICY_CKPT}" >&2
  exit 1
fi

if [[ ! -f "${DICE_CKPT}" ]]; then
  echo "[visualize_lpb_dice] Missing LPB Dice checkpoint: ${DICE_CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[visualize_lpb_dice] ROOT=${ROOT}"
echo "[visualize_lpb_dice] policy.ckpt=${POLICY_CKPT}"
echo "[visualize_lpb_dice] model.dice_ckpt=${DICE_CKPT}"
echo "[visualize_lpb_dice] visualization.data_source=${DATA_SOURCE}"
echo "[visualize_lpb_dice] support_penalty.weight=${SUPPORT_PENALTY_WEIGHT}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_dice/visualize_failures.py" \
  seed="${SEED}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${POLICY_CKPT}" \
  policy.encoder_batch_size="${ENCODER_BATCH_SIZE}" \
  data.image_size="${IMAGE_SIZE}" \
  model.dice_ckpt="${DICE_CKPT}" \
  visualization.num_videos="${NUM_VIDEOS}" \
  visualization.fps="${FPS}" \
  visualization.camera_name="${CAMERA_NAME}" \
  visualization.save_pdf="${SAVE_PDF}" \
  visualization.num_plot_frames="${NUM_PLOT_FRAMES}" \
  visualization.data_source="${DATA_SOURCE}" \
  suboptimal.source_split="${SOURCE_SPLIT}" \
  support_penalty.weight="${SUPPORT_PENALTY_WEIGHT}" \
  "$@"
