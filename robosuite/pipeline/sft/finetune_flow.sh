#!/usr/bin/env bash
set -euo pipefail

cd ~/Documents/DAggar/robosuite

export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=0

CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"
ENVIRONMENT="NutAssemblySquare"
PRETRAIN_SPLIT="pretrain_data-20260615_174814"
PRETRAIN_TRAJ=30
EXPERT=10
POSTFIX="add-expert-10"
OUTPUT_ROOT="./outputs/flow-sft"

STEPS=5000
SAVE_INTERVAL=2500

python -m robosuite.pipeline.sft.flow_sft \
  env.environment="${ENVIRONMENT}" \
  runtime.init_checkpoint="${CKPT}" \
  data.pretrain_split_name="${PRETRAIN_SPLIT}" \
  data.sft_data.expert="${EXPERT}" \
  data.sft_data.pretrain_data="${PRETRAIN_TRAJ}" \
  data.sft_data.success_rollout=0 \
  data.sft_data.fail_rollout=0 \
  train.steps="${STEPS}" \
  train.save_interval="${SAVE_INTERVAL}" \
  output.root="${OUTPUT_ROOT}" \
  output.postfix="${POSTFIX}"
