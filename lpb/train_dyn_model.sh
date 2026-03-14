#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="offline"
export WANDB_ENTITY="songgao-personal"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate lpb

DEFAULT_POLICY_CKPT="/home/dodo/Documents/DAggar/robosuite/lpb/data/outputs/2026.03.09/11.53.10_train_diffusion_unet_hybrid_transport_image/checkpoints/20.ckpt"

ENV_NAME="transport"
TRAIN_DATA_PATH="/home/dodo/Documents/DAggar/robosuite/data/lpb/transport/transport_rollout_and_demo.hdf5"
VAL_DATA_PATH="/home/dodo/Documents/DAggar/robosuite/data/lpb/transport/transport_val.hdf5"
POLICY_CKPT_PATH="${DEFAULT_POLICY_CKPT}"

SEED="42"
DEVICE="cuda:0"
EPOCHS="200"
BATCH_SIZE="128"
SAVE_EVERY="10"
ENCODER_LR=1e-7
PREDICTOR_LR=5e-4
ACTION_ENCODER_LR=5e-4
HYDRA_RUN_DIR='./data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${env.name}'

if [ -z "$POLICY_CKPT_PATH" ]; then
  echo "policy checkpoint is required. Set POLICY_CKPT_PATH=/path/to/base_policy.ckpt"
  exit 1
fi
if [ ! -f "$TRAIN_DATA_PATH" ]; then
  echo "train dataset file not found: $TRAIN_DATA_PATH"
  exit 1
fi
if [ ! -f "$VAL_DATA_PATH" ]; then
  echo "validation dataset file not found: $VAL_DATA_PATH"
  exit 1
fi
if [ ! -f "$POLICY_CKPT_PATH" ]; then
  echo "policy checkpoint file not found: $POLICY_CKPT_PATH"
  exit 1
fi

echo "TRAIN_DATA_PATH=$TRAIN_DATA_PATH"
echo "VAL_DATA_PATH=$VAL_DATA_PATH"
echo "POLICY_CKPT_PATH=$POLICY_CKPT_PATH"
echo "DEVICE=$DEVICE"
echo "SEED=$SEED"
echo "EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE"

cd "$SCRIPT_DIR"
python -X faulthandler -m dyn_model.train \
  --config-name=train \
  env="$ENV_NAME" \
  env.train_data_path="$TRAIN_DATA_PATH" \
  env.val_data_path="$VAL_DATA_PATH" \
  env.policy_ckpt_path="$POLICY_CKPT_PATH" \
  training.seed="$SEED" \
  training.epochs="$EPOCHS" \
  training.batch_size="$BATCH_SIZE" \
  training.save_every_x_epoch="$SAVE_EVERY" \
  training.encoder_lr="$ENCODER_LR" \
  training.predictor_lr="$PREDICTOR_LR" \
  training.action_encoder_lr="$ACTION_ENCODER_LR" \
  hydra.run.dir="$HYDRA_RUN_DIR"
