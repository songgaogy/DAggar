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
#   NUM_TRAJECTORIES  — cap on expert trajectories (default unset = all).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"

PY="${PY:-$HOME/miniconda3/envs/daggar/bin/python}"

ENVIRONMENT="${ENVIRONMENT:-PickPlaceBread}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
LPB_CKPT="${LPB_CKPT:-${ROOT_DIR}/checkpoints/lpb_v2/bce_viz_robosuite/viz_bce_PickPlaceBread-20260518_014008/checkpoints/bce_head.pth}"

VALUE_STEPS="${VALUE_STEPS:-20000}"
FULL_STEPS="${FULL_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-cuda:1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/DIPOLE_RL/iql_qv_cache/${ENVIRONMENT}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/iql_state.pt}"
mkdir -p "${OUTPUT_DIR}"

# IQL warmup runs offline only — no display needed; force egl.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-songgao-personal}"

HYDRA_OVERRIDES=(
  "env.environment=${ENVIRONMENT}"
  "runtime.init_checkpoint=${INIT_CHECKPOINT}"
  "algorithm.discriminator.warm_start_ckpt=${LPB_CKPT}"
  "algorithm.q_learning.warmup_value_steps=${VALUE_STEPS}"
  "algorithm.q_learning.warmup_full_steps=${FULL_STEPS}"
  "algorithm.q_learning.config.device=${DEVICE}"
  "+warmup.output_path=${OUTPUT_FILE}"
  "+warmup.batch_size=${BATCH_SIZE}"
)

if [[ -n "${NUM_TRAJECTORIES:-}" ]]; then
  HYDRA_OVERRIDES+=("data.num_trajectories=${NUM_TRAJECTORIES}")
fi

echo "[init_iql_qv] env=${ENVIRONMENT} device=${DEVICE}"
echo "[init_iql_qv] value_steps=${VALUE_STEPS} full_steps=${FULL_STEPS} batch=${BATCH_SIZE}"
echo "[init_iql_qv] output=${OUTPUT_FILE}"
echo "[init_iql_qv] init_checkpoint=${INIT_CHECKPOINT}"
echo "[init_iql_qv] lpb_ckpt=${LPB_CKPT}"

"${PY}" -m robosuite.pipeline.algorithms.q_learning.warmup "${HYDRA_OVERRIDES[@]}"

echo "[init_iql_qv] done. Reuse with:"
echo "  IQL_WARMUP_CKPT=${OUTPUT_FILE} bash robosuite/pipeline/scripts/train_dipole_rl.sh"
