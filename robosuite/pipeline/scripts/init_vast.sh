#!/usr/bin/env bash
# Standalone VAST warmup utility. This is not part of run_online/train.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
TASK="${TASK:-Stack}"
DEVICE="cuda:0"
STEPS=40000
BATCH_SIZE=512
OUTPUT_PATH="outputs/dipole_rl-vast/tau0p7_en5_disc-reward0p1"

NUM_TRAJECTORIES_SUCCESS="${NUM_TRAJECTORIES_SUCCESS:-50}"
NUM_TRAJECTORIES_FAIL="${NUM_TRAJECTORIES_FAIL:-50}"
SAVE_DATA="${SAVE_DATA:-true}"
SAVE_DIR="${SAVE_DIR:-offline_data-vast}"
OUTPUT_DIR="${OUTPUT_PATH}/${TASK}"
OUTPUT_FILE="${OUTPUT_DIR}/vast_state.pt"
TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"

cd "$ROOT_DIR"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

"$PY" -m robosuite.pipeline.algorithms.vast.warmup \
  "task=${TASK}" \
  "cuda.training_device=${DEVICE}" \
  "task.vast.num_steps=${STEPS}" \
  "+warmup.output_path=${OUTPUT_FILE}" \
  "+warmup.tensorboard_dir=${TENSORBOARD_DIR}" \
  "+warmup.batch_size=${BATCH_SIZE}" \
  "+warmup.demo_splits=[success_rollout,fail_rollout]" \
  "+warmup.num_trajectories.success_rollout=${NUM_TRAJECTORIES_SUCCESS}" \
  "+warmup.num_trajectories.fail_rollout=${NUM_TRAJECTORIES_FAIL}" \
  "+warmup.num_trajectories.save_data=${SAVE_DATA}" \
  "+warmup.num_trajectories.save_dir=${SAVE_DIR}" \
  'hydra.run.dir=.' \
  'hydra.output_subdir=null' \
  'hydra.job.chdir=false'
