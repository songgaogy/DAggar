#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/dodo/Documents/DAggar/robosuite"
PYTHON_BIN="/home/dodo/miniconda3/envs/daggar/bin/python"

export CUDA_VISIBLE_DEVICES=0
export WANDB_NAME="songgao-personal"
export WANDB_MODE="offline"

cd "${ROOT_DIR}"

# contrastive learning objective
exec "${PYTHON_BIN}" -m robosuite.discriminator.lpb.train_dynamics \
  data.expert_paths='["/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert"]' \
  data.rollout_paths='["/home/dodo/Documents/DAggar/robosuite/data/PandaLift/fail_rollout"]' \
  data.camera_name=agentview \
  data.horizon=1 \
  data.image_size=224 \
  data.failure_tail_ratio=0.3 \
  data.failure_soft_start_ratio=0.55 \
  data.failure_soft_power=2.0 \
  encoder.ckpt_path=/home/dodo/Documents/DAggar/robosuite/robosuite/RL/models/resnet18-f37072fd.pth \
  model.d_model=512 \
  model.num_layers=6 \
  model.num_heads=8 \
  model.dropout=0.1 \
  model.max_action_horizon=32 \
  model.fusion_hidden_dim=768 \
  model.projection_dim=0 \
  training.device=cuda \
  training.batch_size=256 \
  training.num_workers=24 \
  training.epochs=100 \
  training.lr=1e-4 \
  training.weight_decay=1e-4 \
  training.expert_ratio=0.5 \
  training.proprio_loss_weight=0.0 \
  training.contrastive_loss_weight=0.8 \
  training.contrastive_temperature=0.2 \
  training.contrastive_negative_confidence_threshold=0.2 \
  training.contrastive_queue_size=4096 \
  training.grad_clip_norm=1.0 \
  training.log_every=50 \
  save_dir=./checkpoints/PandaLift/discriminator/lpb_dynamics/contrastive-v2 \
  save_name=dynamics_model_pu_contrastive_aligned.pt \
  "$@"
