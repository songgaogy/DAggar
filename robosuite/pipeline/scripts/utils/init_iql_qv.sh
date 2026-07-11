#!/usr/bin/env bash
# IQL Q/V offline warmup entry. Produces a per-task `iql_state.pt` that
# `train_dipole_rl.sh` can pick up via IQL_WARMUP_CKPT (which maps to the
# Hydra key `algorithm.q_learning.warmup_ckpt`).
#
# Experiment constants:
#   ENVIRONMENT and INIT_CHECKPOINT are intentionally set below. Edit them for
#   a different task or base policy.
# Required env vars:
#   NNPU_CKPT         — task-calibrated pu_bce_head.pth.
# Optional env vars:
#   SEED              — RNG seed for reproducible warmup (default 42). Same SEED +
#                        same machine/GPU => identical demo-load order and sampling.
#   VALUE_STEPS       — V-only warmup loop length (default 20000).
#   FULL_STEPS        — full IQL update loop length (default 10000).
#   BATCH_SIZE        — minibatch size (default 128).
#   DEVICE            — learner device (default cuda:1).
#   OUTPUT_DIR        — per-task output dir (default outputs/DIPOLE_rl/iql_qv_cache/<ENVIRONMENT>).
#   NUM_TRAJECTORIES  — alias for NUM_TRAJECTORIES_EXPERT (legacy).
#   NUM_TRAJECTORIES_EXPERT / _SUCCESS / _FAIL — per-split HDF5 caps
#                        (unset = config defaults; <=0 = all).
#   WARMUP_DEMO_SPLITS — comma-separated HDF5 subdirs under data/<task>/
#                        (default: expert,success_rollout,fail_rollout).
#   SAVE_DATA         — "true"/"false": save assembled offline transitions to
#                        <data_root>/<task>/<SAVE_DIR>/ (unset = config default).
#   SAVE_DIR          — subdir name for the saved offline data (default offline_data).
#   TENSORBOARD_DIR   — tensorboard event dir (default <OUTPUT_DIR>/tensorboard).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=1

# -------------------------------------
ENVIRONMENT="PickPlaceCereal"
NAME="offline_iql_qv-v2-disc0p02"
BRANCH_NAME="dipole_rl-iql"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth"
SEED=42
# -------------------------------------

DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda:0}"
# Where to hold the pre-encoded chunk cache: "cpu" (default; no VRAM growth) or
# "learner"/"device" (cache on the IQL GPU — faster sampling but OOMs on large
# datasets since it stores one ~14k-float feature row per valid chunk in VRAM).
PREENCODE_CACHE_DEVICE="${PREENCODE_CACHE_DEVICE:-cpu}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${BRANCH_NAME}/${NAME}/${ENVIRONMENT}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/iql_state.pt}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
INIT_CHECKPOINT="checkpoints/multitask_6/flow_multi_ep0100.pt"
mkdir -p "${OUTPUT_DIR}"

# IQL warmup is offline (HDF5 images + proprio); sparse r_env is replayed through
# the robosuite env per step (actual task success), not HDF5 split/last-frame labels.
# Env is built without offscreen rendering.
# MUJOCO_GL is unused unless you override warmup to enable rendering.
export MUJOCO_GL="egl"
export HYDRA_FULL_ERROR=1
# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

HYDRA_OVERRIDES=(
  "seed=${SEED}"
  "env.environment=${ENVIRONMENT}"
  "data.task_name=${DEMO_TASK_NAME}"
  "runtime.init_checkpoint=${INIT_CHECKPOINT}"
  "algorithm.discriminator.checkpoint=${NNPU_CKPT}"
  "algorithm.q_learning.config.device=${DEVICE}"
  "+warmup.output_path=${OUTPUT_FILE}"
  "+warmup.tensorboard_dir=${TENSORBOARD_DIR}"
  "+warmup.batch_size=${BATCH_SIZE}"
  "warmup.preencode_cache_device=${PREENCODE_CACHE_DEVICE}"
)

# Save assembled offline transitions for future reuse (offline DIPOLE etc.).
HYDRA_OVERRIDES+=("warmup.num_trajectories.save_data=true")
HYDRA_OVERRIDES+=("warmup.num_trajectories.save_dir=offline_data-iql")

# Collapse each success demo's post-success drift into one frozen absorbing
# (s, a) anchor (see warmup.freeze_post_success). Default on; set
# FREEZE_POST_SUCCESS=false to keep the raw post-success drift frames.
FREEZE_POST_SUCCESS="${FREEZE_POST_SUCCESS:-true}"
HYDRA_OVERRIDES+=("warmup.freeze_post_success=${FREEZE_POST_SUCCESS}")
HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

echo "[init_iql_qv] env=${ENVIRONMENT} device=${DEVICE} seed=${SEED}"
echo "[init_iql_qv] output=${OUTPUT_FILE}"
echo "[init_iql_qv] tensorboard=${TENSORBOARD_DIR}"
echo "[init_iql_qv] init_checkpoint=${INIT_CHECKPOINT}"
echo "[init_iql_qv] nnpu_ckpt=${NNPU_CKPT}"

"${PY}" -m robosuite.pipeline.algorithms.q_learning.warmup "${HYDRA_OVERRIDES[@]}"

echo "[init_iql_qv] done. Reuse with:"
echo "  IQL_WARMUP_CKPT=${OUTPUT_FILE} bash robosuite/pipeline/scripts/train_dipole_rl.sh"
echo "[init_iql_qv] tensorboard:"
echo "  tensorboard --logdir ${TENSORBOARD_DIR}"
