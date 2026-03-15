#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="offline"
export WANDB_ENTITY="songgao-personal"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export LPB_MUJOCO_PY_STUB="${LPB_MUJOCO_PY_STUB:-1}"
export LPB_RENDER_GPU_IDS="${LPB_RENDER_GPU_IDS:-$CUDA_VISIBLE_DEVICES}"
export LPB_VECTOR_ENV_MODE="${LPB_VECTOR_ENV_MODE:-auto}"
export LPB_VECTOR_ENV_CONTEXT="${LPB_VECTOR_ENV_CONTEXT:-spawn}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate lpb

DATASET_PATH="${DATASET_PATH:-/home/dodo/Documents/DAggar/robosuite/data/lpb/transport/transport_rollout_and_demo.hdf5}"
if [ ! -f "$DATASET_PATH" ]; then
  echo "dataset file not found: $DATASET_PATH"
  exit 1
fi

N_ENVS="${N_ENVS:-8}"
N_TRAIN="${N_TRAIN:-4}"
N_TEST="${N_TEST:-4}"
TRAIN_DEVICE="cuda:0"
ROLLOUT_EVERY=10


python -X faulthandler /home/dodo/Documents/DAggar/robosuite/lpb/train.py \
    --config-dir=. \
    --config-name=image_transport_diffusion_policy_cnn.yaml \
    training.seed=42 \
    training.device="$TRAIN_DEVICE" \
    training.rollout_every="$ROLLOUT_EVERY" \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
    task.dataset_path="$DATASET_PATH" \
    task.env_runner.n_envs="$N_ENVS" \
    task.env_runner.n_train="$N_TRAIN" \
    task.env_runner.n_test="$N_TEST"
