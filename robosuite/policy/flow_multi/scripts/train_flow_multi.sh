#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/dodo/Documents/DAggar/robosuite"
NUM_TRAJ=20
SAVE_DIR="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/policy/flow-${NUM_TRAJ}-new"

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="disabled"
export WANDB_ENTITY="songgao-personal"

# may increase batch_size
python "$ROOT/robosuite/policy/flow_multi/train_flow.py" \
  data.data_dirs='["/home/dodo/Documents/DAggar/robosuite/data/PickPlaceBread/expert_pretrain_data","/home/dodo/Documents/DAggar/robosuite/data/PickPlaceCereal/expert_pretrain_data","/home/dodo/Documents/DAggar/robosuite/data/PickPlaceMilk/expert_pretrain_data","/home/dodo/Documents/DAggar/robosuite/data/PandaPickPlaceCan/expert_pretrain_data","/home/dodo/Documents/DAggar/robosuite/data/PandaStack/expert_pretrain_data","/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert_pretrain_data"]' \
  data.camera_names='["agentview","robot0_robotview","robot0_eye_in_hand"]' \
  data.num_traj=$NUM_TRAJ \
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
  train.device="cuda" \
  checkpoint.save_freq=100
