#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export LPB_RENDER_GPU_IDS="${LPB_RENDER_GPU_IDS:-$CUDA_VISIBLE_DEVICES}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate lpb

CHECKPOINT_DIR="/home/dodo/Documents/DAggar/robosuite/lpb/data/outputs/2026.03.09/11.53.10_train_diffusion_unet_hybrid_transport_image/checkpoints"
EXPERT_DATASET="/home/dodo/Documents/DAggar/robosuite/data/lpb/transport/transport_rollout_and_demo.hdf5"
OUTPUT_DIR="/home/dodo/Documents/DAggar/robosuite/data/lpb/transport/rollouts"
DEVICE="cuda:0"
N_TEST="100"
TEST_START_SEED="100000"
MAX_STEPS="${MAX_STEPS:-}"
OVERWRITE="${OVERWRITE:-0}"
MAX_PARALLEL=4

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "checkpoint directory not found: $CHECKPOINT_DIR"
  exit 1
fi
if [ ! -f "$EXPERT_DATASET" ]; then
  echo "expert dataset file not found: $EXPERT_DATASET"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

mapfile -t CHECKPOINT_PATHS < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name '*.ckpt' | sort -V)
if [ "${#CHECKPOINT_PATHS[@]}" -eq 0 ]; then
  echo "no checkpoints found under: $CHECKPOINT_DIR"
  exit 1
fi

VALID_CHECKPOINT_PATHS=()
SKIPPED_CHECKPOINTS=()
for checkpoint_path in "${CHECKPOINT_PATHS[@]}"; do
  if python - "$checkpoint_path" <<'PY' >/dev/null 2>&1
import sys
import dill
import torch

with open(sys.argv[1], "rb") as f:
    torch.load(f, pickle_module=dill, map_location="cpu")
PY
  then
    VALID_CHECKPOINT_PATHS+=("$checkpoint_path")
  else
    echo "skip invalid checkpoint: $checkpoint_path"
    SKIPPED_CHECKPOINTS+=("$checkpoint_path")
  fi
done

if [ "${#VALID_CHECKPOINT_PATHS[@]}" -eq 0 ]; then
  echo "no valid checkpoints found under: $CHECKPOINT_DIR"
  exit 1
fi

GPU_LIST_RAW="${LPB_RENDER_GPU_IDS:-$CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a GPU_LIST <<< "$GPU_LIST_RAW"
if [ "${#GPU_LIST[@]}" -eq 0 ] || [ -z "${GPU_LIST[0]}" ]; then
  GPU_LIST=("0")
fi

pids=()
pid_names=()
failed_jobs=()
running_jobs=0
job_index=0

wait_for_one_job() {
  local pid
  local checkpoint_name
  pid="${pids[0]}"
  checkpoint_name="${pid_names[0]}"
  if ! wait "$pid"; then
    failed_jobs+=("$checkpoint_name")
  fi
  pids=("${pids[@]:1}")
  pid_names=("${pid_names[@]:1}")
  running_jobs=$((running_jobs - 1))
}

echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "EXPERT_DATASET=$EXPERT_DATASET"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "N_TEST=$N_TEST TEST_START_SEED=$TEST_START_SEED"
echo "MAX_PARALLEL=$MAX_PARALLEL"
echo "CHECKPOINT_COUNT=${#CHECKPOINT_PATHS[@]}"
echo "VALID_CHECKPOINT_COUNT=${#VALID_CHECKPOINT_PATHS[@]}"
echo "SKIPPED_CHECKPOINT_COUNT=${#SKIPPED_CHECKPOINTS[@]}"

cd "$SCRIPT_DIR"
for checkpoint_path in "${VALID_CHECKPOINT_PATHS[@]}"; do
  while [ "$running_jobs" -ge "$MAX_PARALLEL" ]; do
    wait_for_one_job
  done

  gpu_index=$((job_index % ${#GPU_LIST[@]}))
  gpu_id="${GPU_LIST[$gpu_index]}"

  cmd=(
    python -X faulthandler /home/dodo/Documents/DAggar/robosuite/lpb/generate_rollout_dataset.py
    --checkpoint-path "$checkpoint_path"
    --expert-dataset "$EXPERT_DATASET"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --n-test "$N_TEST"
    --test-start-seed "$TEST_START_SEED"
  )

  if [ -n "$MAX_STEPS" ]; then
    cmd+=(--max-steps "$MAX_STEPS")
  fi
  if [ "$OVERWRITE" = "1" ]; then
    cmd+=(--overwrite)
  fi

  echo "launch checkpoint=$(basename "$checkpoint_path") gpu=$gpu_id"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
  LPB_RENDER_GPU_IDS="$gpu_id" \
  "${cmd[@]}" &

  pids+=("$!")
  pid_names+=("$(basename "$checkpoint_path")")
  running_jobs=$((running_jobs + 1))
  job_index=$((job_index + 1))
done

while [ "${#pids[@]}" -gt 0 ]; do
  wait_for_one_job
done

if [ "${#SKIPPED_CHECKPOINTS[@]}" -gt 0 ]; then
  echo "Skipped invalid checkpoints:"
  for checkpoint_path in "${SKIPPED_CHECKPOINTS[@]}"; do
    echo "  - $(basename "$checkpoint_path")"
  done
fi

if [ "${#failed_jobs[@]}" -gt 0 ]; then
  echo "Failed rollout jobs:"
  for checkpoint_name in "${failed_jobs[@]}"; do
    echo "  - $checkpoint_name"
  done
  exit 1
fi
