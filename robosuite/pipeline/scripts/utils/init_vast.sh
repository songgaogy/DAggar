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
#   VAST_V_MODE       — single_vast (default) or ensemble_lcb.
#   VAST_MAX_K        — maximum future distance in macro chunks (default 10).
#   VAST_COMP_COEF    — composition-loss coefficient (default 0.5).
#   EXPECTILE_TAU     — V expectile target (default 0.9).
#   G_LR / V_LR       — G and V learning rates (default 1e-4).
#   OUTPUT_REWARD_COEF / DISC_REWARD_COEF — reward mixture (default 1 / 0).
#   OUTPUT_DIR        — per-task output dir (default outputs/<branch>/<name>/<task>).
#   NUM_TRAJECTORIES  — alias for NUM_TRAJECTORIES_EXPERT (legacy).
#   NUM_TRAJECTORIES_EXPERT / _SUCCESS / _FAIL — per-split HDF5 caps
#                        (unset = config defaults; <=0 = all).
#   WARMUP_DEMO_SPLITS — comma-separated HDF5 subdirs under data/<task>/
#                        (unset = Hydra config defaults).
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
NAME="tau0p7"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth"
SEED="${SEED:-42}"

EXPECTILE_TAU=0.7
VAST_V_MODE="single_vast"   # ensemble_lcb or single_vast
V_ENSEMBLE_SIZE=5
# -------------------------------------

DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cuda:0}"
JOINT_STEPS="${JOINT_STEPS:-40000}"
PREENCODE_CACHE_DEVICE="${PREENCODE_CACHE_DEVICE:-learner}"
PREFER_HDF5_SUCCESS_LABELS="${PREFER_HDF5_SUCCESS_LABELS:-true}"
BULK_READ_HDF5_IMAGES="${BULK_READ_HDF5_IMAGES:-true}"
# VAST defaults. Ensemble parameters are active only in ensemble_lcb mode.

VAST_MAX_K="${VAST_MAX_K:-10}"
VAST_COMP_COEF="${VAST_COMP_COEF:-0.5}"
VAST_SAMPLING_SEED="${VAST_SAMPLING_SEED:-${SEED}}"
G_LR="${G_LR:-1.0e-4}"
V_LR="${V_LR:-1.0e-4}"
OUTPUT_REWARD_COEF="${OUTPUT_REWARD_COEF:-1}"
DISC_REWARD_COEF="${DISC_REWARD_COEF:-0}"
WARMUP_DEMO_SPLITS="${WARMUP_DEMO_SPLITS:-}"
NUM_TRAJECTORIES_EXPERT="${NUM_TRAJECTORIES_EXPERT:-${NUM_TRAJECTORIES:-}}"
NUM_TRAJECTORIES_SUCCESS="${NUM_TRAJECTORIES_SUCCESS:-}"
NUM_TRAJECTORIES_FAIL="${NUM_TRAJECTORIES_FAIL:-}"
SAVE_DATA="${SAVE_DATA:-true}"
SAVE_DIR="${SAVE_DIR:-offline_data-vast}"

LCB_BETA="${LCB_BETA:-0.5}"
BOOTSTRAP_PROB="${BOOTSTRAP_PROB:-0.5}"
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
  "algorithm.vast.config.vast_max_k=${VAST_MAX_K}"
  "algorithm.vast.config.vast_comp_coef=${VAST_COMP_COEF}"
  "algorithm.vast.config.vast_sampling_seed=${VAST_SAMPLING_SEED}"
  "algorithm.vast.config.expectile_tau=${EXPECTILE_TAU}"
  "algorithm.vast.config.g_lr=${G_LR}"
  "algorithm.vast.config.v_lr=${V_LR}"
  "algorithm.vast.config.output_reward_coef=${OUTPUT_REWARD_COEF}"
  "algorithm.vast.config.disc_reward_coef=${DISC_REWARD_COEF}"
  "algorithm.vast.config.v_ensemble_size=${V_ENSEMBLE_SIZE}"
  "algorithm.vast.config.ensemble_lcb_beta=${LCB_BETA}"
  "algorithm.vast.config.ensemble_bootstrap_prob=${BOOTSTRAP_PROB}"
  "+warmup.output_path=${OUTPUT_FILE}"
  "+warmup.tensorboard_dir=${TENSORBOARD_DIR}"
  "+warmup.batch_size=${BATCH_SIZE}"
  "algorithm.vast.warmup_joint_steps=${JOINT_STEPS}"
  "warmup.preencode_cache_device=${PREENCODE_CACHE_DEVICE}"
  "warmup.prefer_hdf5_success_labels=${PREFER_HDF5_SUCCESS_LABELS}"
  "warmup.bulk_read_hdf5_images=${BULK_READ_HDF5_IMAGES}"
)

# Optional data-selection overrides. Empty values preserve the Hydra defaults.
if [[ -n "${WARMUP_DEMO_SPLITS}" ]]; then
  IFS=',' read -r -a DEMO_SPLIT_ITEMS <<< "${WARMUP_DEMO_SPLITS}"
  DEMO_SPLIT_LIST="[$(IFS=','; echo "${DEMO_SPLIT_ITEMS[*]}")]"
  HYDRA_OVERRIDES+=("warmup.demo_splits=${DEMO_SPLIT_LIST}")
fi
if [[ -n "${NUM_TRAJECTORIES_EXPERT}" ]]; then
  HYDRA_OVERRIDES+=("warmup.num_trajectories.expert=${NUM_TRAJECTORIES_EXPERT}")
fi
if [[ -n "${NUM_TRAJECTORIES_SUCCESS}" ]]; then
  HYDRA_OVERRIDES+=("warmup.num_trajectories.success_rollout=${NUM_TRAJECTORIES_SUCCESS}")
fi
if [[ -n "${NUM_TRAJECTORIES_FAIL}" ]]; then
  HYDRA_OVERRIDES+=("warmup.num_trajectories.fail_rollout=${NUM_TRAJECTORIES_FAIL}")
  HYDRA_OVERRIDES+=("warmup.num_trajectories.fail_rollout-labeled=${NUM_TRAJECTORIES_FAIL}")
fi

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
echo "[init_vast] algorithm=vast_value_stitching_adaptation v_mode=${VAST_V_MODE} K=${VAST_MAX_K} comp_coef=${VAST_COMP_COEF} joint_steps=${JOINT_STEPS}"
echo "[init_vast] value: expectile_tau=${EXPECTILE_TAU} g_lr=${G_LR} v_lr=${V_LR} v_ensemble_size=${V_ENSEMBLE_SIZE} lcb_beta=${LCB_BETA} bootstrap_prob=${BOOTSTRAP_PROB}"
echo "[init_vast] reward: output_coef=${OUTPUT_REWARD_COEF} disc_coef=${DISC_REWARD_COEF} sampling_seed=${VAST_SAMPLING_SEED}"
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
