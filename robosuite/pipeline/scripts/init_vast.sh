#!/usr/bin/env bash
# Standalone VAST warmup utility. This is not part of run_online/train.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
TASK="PickPlaceCereal"
DEVICE="cuda:0"
STEPS=40000
BATCH_SIZE=512
OUTPUT_PATH=""

TENSORBOARD_DIR="$(dirname "$OUTPUT_PATH")/tensorboard"
cd "$ROOT_DIR"
export HYDRA_FULL_ERROR=1

"$PY" -m robosuite.pipeline.algorithms.vast.warmup \
  "task=${TASK}" \
  "cuda.training_device=${DEVICE}" \
  "task.vast.num_steps=${STEPS}" \
  "+warmup.output_path=${OUTPUT_PATH}" \
  "+warmup.tensorboard_dir=${TENSORBOARD_DIR}" \
  "+warmup.batch_size=${BATCH_SIZE}" \
  'hydra.run.dir=.' \
  'hydra.output_subdir=null' \
  'hydra.job.chdir=false'
