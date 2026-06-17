#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/dodo/Documents/DAggar/robosuite"
PYTHON_BIN="/home/dodo/miniconda3/envs/dagger/bin/python"
CKPT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/flow-20/flow_multi_ep0100_20260320_114720.pt"
NUM_TRAJS=100
DEVICE="cuda"
MAX_STEPS=400
N_ODE_STEPS="${N_ODE_STEPS:-20}"
ACTION_HORIZON="${ACTION_HORIZON:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
RENDER_HEIGHT="${RENDER_HEIGHT:-256}"
RENDER_WIDTH="${RENDER_WIDTH:-256}"
CUDA_DEVICES=0
MAX_PARALLEL_JOBS=6
LOG_DIR="/home/dodo/Documents/DAggar/robosuite/logs/generate_flow_multi_rollout"

mkdir -p "$LOG_DIR"
TASK_NAMES=("$@")
if [ ${#TASK_NAMES[@]} -eq 0 ]; then
  TASK_NAMES=(
    PickPlaceBread
    PickPlaceCereal
    PickPlaceMilk
    PickPlaceCan
    Stack
    Lift
  )
fi

IFS=',' read -r -a GPU_ARRAY <<< "$CUDA_DEVICES"
if [ ${#GPU_ARRAY[@]} -eq 0 ] || [ -z "${GPU_ARRAY[0]}" ]; then
  GPU_ARRAY=("0")
fi

launch_index=0
running_jobs=0

launch_job() {
  local task_name="$1"
  local keep_mode="$2"
  local gpu="${GPU_ARRAY[$((launch_index % ${#GPU_ARRAY[@]}))]}"
  local log_path="$LOG_DIR/${task_name}_${keep_mode}.log"

  echo "launch task=${task_name} keep_mode=${keep_mode} gpu=${gpu} log=${log_path}"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_GL=egl \
    "$PYTHON_BIN" "$ROOT/robosuite/policy/flow_multi/generate_rollout_data.py" \
    generate.ckpt="$CKPT" \
    generate.task_name="$task_name" \
    generate.output_dir=null \
    generate.num_trajs="$NUM_TRAJS" \
    generate.device="$DEVICE" \
    generate.keep_mode="$keep_mode" \
    generate.max_steps="$MAX_STEPS" \
    generate.n_ode_steps="$N_ODE_STEPS" \
    generate.action_horizon="$ACTION_HORIZON" \
    generate.render_height="$RENDER_HEIGHT" \
    generate.render_width="$RENDER_WIDTH" \
    data.image_size="$IMAGE_SIZE" \
    >"$log_path" 2>&1 &

  launch_index=$((launch_index + 1))
}

for task_name in "${TASK_NAMES[@]}"; do
  for keep_mode in success fail; do
    launch_job "$task_name" "$keep_mode"
    running_jobs=$((running_jobs + 1))
    if [ "$running_jobs" -ge "$MAX_PARALLEL_JOBS" ]; then
      wait -n
      running_jobs=$((running_jobs - 1))
    fi
  done
done

wait

echo "all rollout jobs finished"
echo "logs: $LOG_DIR"
