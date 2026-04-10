#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_dice}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="lpb_dice_${TIMESTAMP}"
SAVE_NAME="${SAVE_NAME:-${RUN_NAME}.pt}"
SAVE_DIR="${SAVE_DIR:-${BASE_SAVE_DIR}/${RUN_NAME}}"
POLICY_CKPT="${POLICY_CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
BACKBONE_CKPT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/lpb_new/lpb_new_20260324_232921/lpb_new_20260324_232921.pt"

BATCH_SIZE=1024
NUM_WORKERS=16
EPOCHS="${EPOCHS:-50}"
LR="${LR:-2e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
HORIZON="${HORIZON:-1}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-96}"

DETECTOR_BATCH_SIZE=1024
DETECTOR_EPOCHS=30
DETECTOR_LR="${DETECTOR_LR:-2e-4}"
DETECTOR_HIDDEN_DIM="${DETECTOR_HIDDEN_DIM:-512}"
DETECTOR_NUM_LAYERS="${DETECTOR_NUM_LAYERS:-2}"
SUPPORT_PENALTY_WEIGHT="${SUPPORT_PENALTY_WEIGHT:-0.0}"

# -------------------------------------
TRAIN_EXPERT=100
TRAIN_SUCCESS=100
TRAIN_FAIL=100
EVAL_EXPERT=20
EVAL_SUCCESS=20
EVAL_FAIL=20
# -------------------------------------

if [[ ! -f "${POLICY_CKPT}" ]]; then
  echo "[train_lpb_dice] Missing policy checkpoint: ${POLICY_CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[train_lpb_dice] ROOT=${ROOT}"
echo "[train_lpb_dice] policy.ckpt=${POLICY_CKPT}"
echo "[train_lpb_dice] save_dir=${SAVE_DIR}"
if [[ -n "${BACKBONE_CKPT}" ]]; then
  echo "[train_lpb_dice] model.backbone_ckpt=${BACKBONE_CKPT}"
fi

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_dice/train.py" \
  seed="${SEED}" \
  hydra.run.dir="${SAVE_DIR}" \
  save_name="${SAVE_NAME}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${POLICY_CKPT}" \
  policy.encoder_batch_size="${ENCODER_BATCH_SIZE}" \
  model.backbone_ckpt="${BACKBONE_CKPT}" \
  data.image_size="${IMAGE_SIZE}" \
  data.transition_horizon="${HORIZON}" \
  data.splits.train.num_expert_traj="${TRAIN_EXPERT}" \
  data.splits.train.num_success_traj="${TRAIN_SUCCESS}" \
  data.splits.train.num_fail_traj="${TRAIN_FAIL}" \
  data.splits.eval.num_expert_traj="${EVAL_EXPERT}" \
  data.splits.eval.num_success_traj="${EVAL_SUCCESS}" \
  data.splits.eval.num_fail_traj="${EVAL_FAIL}" \
  training.batch_size="${BATCH_SIZE}" \
  training.num_workers="${NUM_WORKERS}" \
  training.epochs="${EPOCHS}" \
  training.lr="${LR}" \
  detector_training.batch_size="${DETECTOR_BATCH_SIZE}" \
  detector_training.epochs="${DETECTOR_EPOCHS}" \
  detector_training.lr="${DETECTOR_LR}" \
  detector_training.hidden_dim="${DETECTOR_HIDDEN_DIM}" \
  detector_training.num_layers="${DETECTOR_NUM_LAYERS}" \
  support_penalty.weight="${SUPPORT_PENALTY_WEIGHT}" \
  "$@"
