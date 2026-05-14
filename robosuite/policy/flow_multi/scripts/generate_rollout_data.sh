#!/usr/bin/env bash
# Flow-multi rollout HDF5 generation (Hydra: generate_rollout_data.py).
# Edit CONFIG only, then run with no arguments: ./generate_rollout_data.sh
set -euo pipefail

# =============================================================================
# CONFIG - set everything here (no CLI overrides)
# =============================================================================

# Runtime
PYTHON="/home/dodo/miniconda3/envs/daggar/bin/python"
CUDA_VISIBLE_DEVICES="1"
MUJOCO_GL="egl"

# generate.* (mirrors config/generate_rollout_data.yaml)
KEEP_MODE="fail" # success | fail | all
CKPT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt"
TASK_NAME="PickPlaceMilk"
OUTPUT_DIR="/home/dodo/Documents/DAggar/robosuite/data/PickPlaceMilk/fail_rollout_train"       # empty -> Hydra null (Python picks default under data/<task>/...)
NUM_TRAJS=""        # empty -> Hydra null (Python uses EPISODES)
EPISODES="100"
DEVICE="cuda"
MAX_STEPS="400"
N_ODE_STEPS="10"
ACTION_HORIZON="4"
RENDER_HEIGHT="256"
RENDER_WIDTH="256"

# data.*
IMAGE_SIZE="128"

echo "--------------------------------"
echo "TASK_NAME: ${TASK_NAME}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "--------------------------------"

# =============================================================================
# below: wiring only
# =============================================================================

if (($# > 0)); then
  echo "error: this script accepts no arguments; edit CONFIG at the top of:" >&2
  echo "  ${BASH_SOURCE[0]}" >&2
  exit 1
fi

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly FLOW_MULTI="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT="$(cd "${FLOW_MULTI}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES
export MUJOCO_GL

readonly ENTRY="${FLOW_MULTI}/generate_rollout_data.py"

die() { echo "error: $*" >&2; exit 1; }

km="${KEEP_MODE,,}"
case "${km}" in success | fail | all) ;; *) die "KEEP_MODE must be success, fail, or all (got: ${KEEP_MODE})" ;; esac

hydra_output_dir="${OUTPUT_DIR}"
[[ -n "${hydra_output_dir}" ]] || hydra_output_dir="null"

hydra_num_trajs="${NUM_TRAJS}"
[[ -n "${hydra_num_trajs}" ]] || hydra_num_trajs="null"

hydra_overrides=(
  "generate.keep_mode=${km}"
  "generate.ckpt=${CKPT}"
  "generate.task_name=${TASK_NAME}"
  "generate.output_dir=${hydra_output_dir}"
  "generate.num_trajs=${hydra_num_trajs}"
  "generate.episodes=${EPISODES}"
  "generate.device=${DEVICE}"
  "generate.max_steps=${MAX_STEPS}"
  "generate.n_ode_steps=${N_ODE_STEPS}"
  "generate.action_horizon=${ACTION_HORIZON}"
  "generate.render_height=${RENDER_HEIGHT}"
  "generate.render_width=${RENDER_WIDTH}"
  "data.image_size=${IMAGE_SIZE}"
)

exec "${PYTHON}" "${ENTRY}" "${hydra_overrides[@]}"
