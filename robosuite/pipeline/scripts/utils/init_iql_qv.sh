#!/usr/bin/env bash
# IQL Q/V offline warmup entry. Produces a per-task `iql_state.pt` that
# `train_dipole_rl.sh` can pick up via IQL_WARMUP_CKPT (which maps to the
# Hydra key `algorithm.q_learning.warmup_ckpt`).
#
# Required env vars:
#   INIT_CHECKPOINT   — flow-dagger / flow-multi base checkpoint (provides
#                        camera layout + policy action_dim).
# Optional env vars:
#   ENVIRONMENT       — task env name (default PickPlaceBread).
#   LPB_CKPT          — overrides algorithm.discriminator.warm_start_ckpt.
#   VALUE_STEPS       — V-only warmup loop length (default 20000).
#   FULL_STEPS        — full IQL update loop length (default 5000).
#   BATCH_SIZE        — minibatch size (default 64).
#   DEVICE            — learner device (default cuda:1).
#   OUTPUT_DIR        — per-task output dir (default outputs/DIPOLE_RL/iql_qv_cache/<ENVIRONMENT>).
#   NUM_TRAJECTORIES  — alias for NUM_TRAJECTORIES_EXPERT (legacy).
#   NUM_TRAJECTORIES_EXPERT / _SUCCESS / _FAIL — per-split HDF5 caps
#                        (unset = all; maps to warmup.num_trajectories.*).
#   WARMUP_DEMO_SPLITS — comma-separated HDF5 subdirs under data/<task>/
#                        (default: expert,success_rollout,fail_rollout).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"

PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"

ENVIRONMENT="${ENVIRONMENT:-PickPlaceMilk}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
LPB_CKPT="${LPB_CKPT:-${ROOT_DIR}/checkpoints/lpb_v2/bce_viz_robosuite/viz_bce_PickPlaceMilk-20260518_014308/checkpoints/bce_head.pth}"

VALUE_STEPS="${VALUE_STEPS:-20000}"
FULL_STEPS="${FULL_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda:1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/DIPOLE_rl/iql_qv_cache/${ENVIRONMENT}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/iql_state.pt}"
mkdir -p "${OUTPUT_DIR}"

# IQL warmup is offline (HDF5 only); env is built without offscreen rendering.
# MUJOCO_GL is unused unless you override warmup to enable rendering.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-songgao-personal}"
export HYDRA_FULL_ERROR=1

# Match train_dipole.sh: demo HDF5 dirs use env name (e.g. PickPlaceBread),
# not robot-prefixed PandaPickPlaceBread from resolve_demo_task_name fallback.
DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"

HYDRA_OVERRIDES=(
  "env.environment=${ENVIRONMENT}"
  "data.task_name=${DEMO_TASK_NAME}"
  "runtime.init_checkpoint=${INIT_CHECKPOINT}"
  "algorithm.discriminator.warm_start_ckpt=${LPB_CKPT}"
  "algorithm.q_learning.warmup_value_steps=${VALUE_STEPS}"
  "algorithm.q_learning.warmup_full_steps=${FULL_STEPS}"
  "algorithm.q_learning.config.device=${DEVICE}"
  "+warmup.output_path=${OUTPUT_FILE}"
  "+warmup.batch_size=${BATCH_SIZE}"
)

# set number of trajectories for each split
# NOTE: we do not need gt-fail label in iql training
NUM_TRAJECTORIES_EXPERT="${NUM_TRAJECTORIES_EXPERT:-${NUM_TRAJECTORIES:-}}"
if [[ -n "${NUM_TRAJECTORIES_EXPERT:-}" ]]; then
  HYDRA_OVERRIDES+=("warmup.num_trajectories.expert=${NUM_TRAJECTORIES_EXPERT}")
fi
if [[ -n "${NUM_TRAJECTORIES_SUCCESS:-}" ]]; then
  HYDRA_OVERRIDES+=("warmup.num_trajectories.success_rollout=${NUM_TRAJECTORIES_SUCCESS}")
fi
if [[ -n "${NUM_TRAJECTORIES_FAIL:-}" ]]; then
  HYDRA_OVERRIDES+=("warmup.num_trajectories.fail_rollout=${NUM_TRAJECTORIES_FAIL}")
fi

if [[ -n "${WARMUP_DEMO_SPLITS:-}" ]]; then
  IFS=',' read -r -a _warmup_splits <<< "${WARMUP_DEMO_SPLITS}"
  for _split in "${_warmup_splits[@]}"; do
    HYDRA_OVERRIDES+=("+warmup.demo_splits+=${_split}")
  done
fi

if [[ "${DEVICE}" == cuda* ]]; then
  if ! "${PY}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "[ERROR] DEVICE=${DEVICE} but torch.cuda.is_available() is False." >&2
    echo "        Run: nvidia-smi   (fix driver/library mismatch; reboot after driver update)" >&2
    exit 1
  fi
fi

echo "[init_iql_qv] env=${ENVIRONMENT} device=${DEVICE}"
echo "[init_iql_qv] value_steps=${VALUE_STEPS} full_steps=${FULL_STEPS} batch=${BATCH_SIZE}"
echo "[init_iql_qv] output=${OUTPUT_FILE}"
echo "[init_iql_qv] init_checkpoint=${INIT_CHECKPOINT}"
echo "[init_iql_qv] lpb_ckpt=${LPB_CKPT}"

"${PY}" -m robosuite.pipeline.algorithms.q_learning.warmup "${HYDRA_OVERRIDES[@]}"

echo "[init_iql_qv] done. Reuse with:"
echo "  IQL_WARMUP_CKPT=${OUTPUT_FILE} bash robosuite/pipeline/scripts/train_dipole_rl.sh"
