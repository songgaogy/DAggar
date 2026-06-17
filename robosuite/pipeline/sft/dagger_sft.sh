#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/dodo/Documents/DAggar/robosuite}"
PYTHON="${PYTHON:-/home/dodo/miniconda3/envs/dagger/bin/python}"

ENVIRONMENT="PickPlaceCereal"
DEMO_BUFFER="outputs/flow-DAgger/flow_dagger_PickPlaceCereal_2026-06-16_23-28-35_no-update/buffers/demo_chunks/chunk_00000000.pt"
WARMUP_NUM=10
WARMP_UP_SPLIT="pretrain_data-20260615_174814"
CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"

STEPS=5000
SAVE_INTERVAL=2500
DEVICE="cuda:0"
OUTPUT_ROOT="./outputs/DAgger-sft"

LEARNING_RATE="${LEARNING_RATE:-3.0e-6}"
TRAIN_SCOPE="${TRAIN_SCOPE:-all}"
SEED_VALUE="${SEED:-42}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
FREEZE_VISUAL_BN="${FREEZE_VISUAL_BN:-false}"
EMA_DECAY="${EMA_DECAY:-0.999}"
DEMO_ROOT="${DEMO_ROOT:-./data}"
DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"

export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "${ROOT_DIR}"

"${PYTHON}" -m robosuite.pipeline.sft.dagger_sft \
  seed="${SEED_VALUE}" \
  env.environment="${ENVIRONMENT}" \
  data.task_name="${DEMO_TASK_NAME}" \
  data.demo_root="${DEMO_ROOT}" \
  data.demo_buffer="${DEMO_BUFFER}" \
  data.warmup_split_name="${WARMP_UP_SPLIT}" \
  data.warmup_num="${WARMUP_NUM}" \
  runtime.init_checkpoint="${CKPT}" \
  train.steps="${STEPS}" \
  train.save_interval="${SAVE_INTERVAL}" \
  train.device="${DEVICE}" \
  train.learning_rate="${LEARNING_RATE}" \
  train.train_scope="${TRAIN_SCOPE}" \
  train.freeze_visual_bn="${FREEZE_VISUAL_BN}" \
  train.ema_decay="${EMA_DECAY}" \
  train.log_interval="${LOG_INTERVAL}" \
  output.root="${OUTPUT_ROOT}" \
  "$@"
