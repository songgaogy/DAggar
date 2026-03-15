#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE="offline"
export WANDB_ENTITY="songgao-personal"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export ZARR_V3_EXPERIMENTAL_API="${ZARR_V3_EXPERIMENTAL_API:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate lpb

DEFAULT_POLICY_CKPT="/home/dodo/Documents/DAggar/robosuite/lpb/data/outputs/2026.03.09/11.53.10_train_diffusion_unet_hybrid_transport_image/checkpoints/20.ckpt"

ENV_NAME="transport"
EXPERT_DATA_PATH="/home/dodo/Documents/DAggar/robosuite/data/lpb/transport/transport_rollout_and_demo.hdf5"
ROLLOUT_DATA_DIR="/home/dodo/Documents/DAggar/robosuite/data/lpb/transport/rollouts"
VAL_DATA_PATH="/home/dodo/Documents/DAggar/robosuite/data/lpb/transport/transport_val.hdf5"
POLICY_CKPT_PATH="${POLICY_CKPT_PATH:-$DEFAULT_POLICY_CKPT}"

SEED="42"
DEVICE="cuda:0"
EPOCHS="200"
BATCH_SIZE="64"
SAVE_EVERY="10"
NUM_WORKERS=24
PREFETCH_FACTOR=16
PIN_MEMORY="${PIN_MEMORY:-true}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-true}"
ENCODER_LR=1e-7
PREDICTOR_LR=5e-4
ACTION_ENCODER_LR=5e-4
HYDRA_RUN_DIR='/home/dodo/Documents/DAggar/robosuite/lpb/data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${env.name}'

if [ -z "$POLICY_CKPT_PATH" ]; then
  echo "policy checkpoint is required. Set POLICY_CKPT_PATH=/path/to/base_policy.ckpt"
  exit 1
fi
if [ ! -f "$EXPERT_DATA_PATH" ]; then
  echo "expert dataset file not found: $EXPERT_DATA_PATH"
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

TRAIN_DATA_FILES=("$EXPERT_DATA_PATH")
if [ -d "$ROLLOUT_DATA_DIR" ]; then
  while IFS= read -r rollout_file; do
    TRAIN_DATA_FILES+=("$rollout_file")
  done < <(find "$ROLLOUT_DATA_DIR" -maxdepth 1 -type f -name '*.hdf5' | sort)
fi

TRAIN_DATA_PATHS='['
for train_data_file in "${TRAIN_DATA_FILES[@]}"; do
  escaped_path="${train_data_file//\\/\\\\}"
  escaped_path="${escaped_path//\"/\\\"}"
  TRAIN_DATA_PATHS+="\"$escaped_path\","
done
TRAIN_DATA_PATHS="${TRAIN_DATA_PATHS%,}]"

echo "EXPERT_DATA_PATH=$EXPERT_DATA_PATH"
echo "ROLLOUT_DATA_DIR=$ROLLOUT_DATA_DIR"
echo "NUM_TRAIN_DATA_FILES=${#TRAIN_DATA_FILES[@]}"
echo "VAL_DATA_PATH=$VAL_DATA_PATH"
echo "POLICY_CKPT_PATH=$POLICY_CKPT_PATH"
echo "DEVICE=$DEVICE"
echo "SEED=$SEED"
echo "EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE"
echo "NUM_WORKERS=$NUM_WORKERS PREFETCH_FACTOR=$PREFETCH_FACTOR"
echo "PIN_MEMORY=$PIN_MEMORY PERSISTENT_WORKERS=$PERSISTENT_WORKERS"

cd "$SCRIPT_DIR"/..
python -X faulthandler -m dyn_model.train \
  --config-name=train \
  env="$ENV_NAME" \
  env.train_data_path="$TRAIN_DATA_PATHS" \
  env.val_data_path="$VAL_DATA_PATH" \
  env.policy_ckpt_path="$POLICY_CKPT_PATH" \
  training.seed="$SEED" \
  training.epochs="$EPOCHS" \
  training.batch_size="$BATCH_SIZE" \
  training.save_every_x_epoch="$SAVE_EVERY" \
  training.num_workers="$NUM_WORKERS" \
  training.prefetch_factor="$PREFETCH_FACTOR" \
  training.pin_memory="$PIN_MEMORY" \
  training.persistent_workers="$PERSISTENT_WORKERS" \
  training.encoder_lr="$ENCODER_LR" \
  training.predictor_lr="$PREDICTOR_LR" \
  training.action_encoder_lr="$ACTION_ENCODER_LR" \
  hydra.run.dir="$HYDRA_RUN_DIR"
