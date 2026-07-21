#!/usr/bin/env bash
# VAST value-stitching offline warmup entry. Produces a per-task `vast_state.pt`
# that `train_dipole_rl.sh` can pick up via VAST_WARMUP_CKPT (which maps to
# the Hydra key `algorithm.vast.warmup_ckpt`).
#
# Experiment constants:
#   ENVIRONMENT and INIT_CHECKPOINT are intentionally set below. Edit them for
#   a different task or base policy.
# Required env vars:
#   NNPU_CKPT         — task-calibrated pu_bce_head.pth.
# Optional env vars:
#   SEED              — RNG seed for reproducible warmup (default 42). Same SEED +
#                        same machine/GPU => identical demo-load order and sampling.
#   JOINT_STEPS       — joint G+V warmup loop length (default 30000).
#   BATCH_SIZE        — minibatch size (default 512).
#   DEVICE            — learner device (default cuda:0 inside CUDA_VISIBLE_DEVICES=1).
#   VAST_V_MODE       — single_vast (default) or indep_ensemble.
#   EXPECTILE_TAU     — V expectile target (default 0.9).
#   OUTPUT_DIR        — per-task output dir (default outputs/<branch>/<name>/<task>).
#   SAVE_DATA         — "true"/"false": save assembled offline transitions to
#                        <data_root>/<task>/<SAVE_DIR>/ (unset = config default).
#   SAVE_DIR          — subdir name for the saved offline data (default offline_data-vast).
#   TENSORBOARD_DIR   — tensorboard event dir (default <OUTPUT_DIR>/tensorboard).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=0

BRANCH_NAME="dipole_rl-vast"

# -------------------------------------
ENVIRONMENT="PickPlaceCereal"
NAME="tau0p7_en5_disc-reward0p1"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite-chunk_v2-P98/run_20260719_204315_PickPlaceCereal/checkpoints/pu_bce_head.pth"
SEED="${SEED:-42}"

EXPECTILE_TAU=0.7
VAST_V_MODE="indep_ensemble"   # indep_ensemble or single_vast
V_ENSEMBLE_SIZE=5
# -------------------------------------

DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda:0}"
JOINT_STEPS="${JOINT_STEPS:-40000}"
# VAST defaults. Ensemble size is active only in indep_ensemble mode.
SAVE_DATA="${SAVE_DATA:-true}"
SAVE_DIR="${SAVE_DIR:-offline_data-vast}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${BRANCH_NAME}/${NAME}/${ENVIRONMENT}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/vast_state.pt}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/multitask_6/flow_multi_ep0100.pt}"
mkdir -p "${OUTPUT_DIR}"

# VAST warmup is offline (HDF5 images + proprio). By default, rollout HDF5
# `is_success` labels are shifted to post-action rewards; set
# PREFER_HDF5_SUCCESS_LABELS=false to replay sparse r_env through robosuite.
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
  "algorithm.vast.config.device=${DEVICE}"
  "algorithm.vast.config.vast_v_mode=${VAST_V_MODE}"
  "algorithm.vast.config.expectile_tau=${EXPECTILE_TAU}"
  "algorithm.vast.config.v_ensemble_size=${V_ENSEMBLE_SIZE}"
  "+warmup.output_path=${OUTPUT_FILE}"
  "+warmup.tensorboard_dir=${TENSORBOARD_DIR}"
  "+warmup.batch_size=${BATCH_SIZE}"
  "algorithm.vast.warmup_joint_steps=${JOINT_STEPS}"
)

# Save assembled offline transitions for future reuse (offline DIPOLE etc.).
HYDRA_OVERRIDES+=("warmup.num_trajectories.save_data=${SAVE_DATA}")
HYDRA_OVERRIDES+=("warmup.num_trajectories.save_dir=${SAVE_DIR}")

# Collapse each success demo's post-success drift into one frozen absorbing
# (s, a) anchor (see warmup.freeze_post_success). Default on; set
# FREEZE_POST_SUCCESS=false to keep the raw post-success drift frames.
FREEZE_POST_SUCCESS="${FREEZE_POST_SUCCESS:-true}"
HYDRA_OVERRIDES+=("warmup.freeze_post_success=${FREEZE_POST_SUCCESS}")
HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

echo "[init_vast] env=${ENVIRONMENT} device=${DEVICE} seed=${SEED}"
echo "[init_vast] algorithm=vast_value_stitching_adaptation v_mode=${VAST_V_MODE} joint_steps=${JOINT_STEPS}"
echo "[init_vast] value: expectile_tau=${EXPECTILE_TAU} v_ensemble_size=${V_ENSEMBLE_SIZE}"
echo "[init_vast] output=${OUTPUT_FILE}"
echo "[init_vast] offline_buffer_dir=${ROOT_DIR}/data/${DEMO_TASK_NAME}/${SAVE_DIR}"
echo "[init_vast] tensorboard=${TENSORBOARD_DIR}"
echo "[init_vast] init_checkpoint=${INIT_CHECKPOINT}"
echo "[init_vast] nnpu_ckpt=${NNPU_CKPT}"

"${PY}" -m robosuite.pipeline.algorithms.vast.warmup "${HYDRA_OVERRIDES[@]}"

echo "[init_vast] done. Reuse with:"
echo "  VAST_WARMUP_CKPT=${OUTPUT_FILE} bash robosuite/pipeline/scripts/train_dipole_rl.sh"
echo "[init_vast] tensorboard:"
echo "  tensorboard --logdir ${TENSORBOARD_DIR}"
