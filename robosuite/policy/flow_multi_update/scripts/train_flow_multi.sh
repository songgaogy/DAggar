#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3"
SAVE_DIR="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/checkpoints/multitask_6/policy/flow-update"
SAVE_TRAIN_DATA=true

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="disabled"
export WANDB_ENTITY="songgao-personal"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

# may increase batch_size
cd "$ROOT"
"$PYTHON_BIN" -m robosuite.policy.flow_multi_update.train_flow \
  data.save_pretrain_data=$SAVE_TRAIN_DATA \
  data.preload_all_demos_to_ram=true \
  data.use_demo_cache=true \
  data.use_disk_cache=true \
  checkpoint.save_dir="$SAVE_DIR" \
  train.epochs=400 \
  train.batch_size=256 \
  train.num_workers=16 \
  train.prefetch_factor=8 \
  train.lr=1e-4 \
  train.min_lr=1e-6 \
  train.weight_decay=1e-6 \
  train.grad_clip_norm=1.0 \
  train.log_freq=1 \
  train.device="cuda" \
  checkpoint.save_freq=50
