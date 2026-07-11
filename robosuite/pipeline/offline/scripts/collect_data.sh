#!/usr/bin/env bash
# Offline data collection entry. Runs a fixed flow policy with optional
# SpaceMouse intervention and nnPU HUD/scoring, then saves torch episode packs.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
cd "$ROOT_DIR"

# -------------------------------------
TASK="NutAssemblySquare"
POLICY_CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_203140_NutAssemblySquare/checkpoints/pu_bce_head.pth"
NUM_TRAJECTORIES=50
# -------------------------------------

DISC_DEVICE="${DISC_DEVICE:-cuda:0}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda:1}"

INTERACTIVE="${INTERACTIVE:-true}"
VIEWER_ENABLED="${VIEWER_ENABLED:-true}"
INTERVENTION_ENABLED="${INTERVENTION_ENABLED:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-data/${TASK}/offline_data}"
OUTPUT_FILE="${OUTPUT_FILE:-offline_episodes.pt}"
EPISODE_MAX_STEPS="${EPISODE_MAX_STEPS:-500}"
DETERMINISTIC="${DETERMINISTIC:-false}"
SAVE_EVERY_EPISODE="${SAVE_EVERY_EPISODE:-false}"
SAVE_INTERVAL_EPISODES="${SAVE_INTERVAL_EPISODES:-0}"
IMAGE_OBS_FPS="${IMAGE_OBS_FPS:-20}"
POLICY_FPS="${POLICY_FPS:-20}"
SPACEMOUSE_FPS="${SPACEMOUSE_FPS:-20}"
DISC_INFERENCE_FPS="${DISC_INFERENCE_FPS:--1}"
DISC_INTERVENE_ENV="${DISC_INTERVENE_ENV:-true}"
SEED_VALUE="${SEED:-42}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-4}"
N_ODE_STEPS="${N_ODE_STEPS:-10}"

export HYDRA_FULL_ERROR=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

if [[ "${INTERACTIVE}" == "true" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

HYDRA_OVERRIDES=(
  "seed=${SEED_VALUE}"
  "env.environment=${TASK}"
  "env.renderer=mjviewer"
  "runtime.init_checkpoint=${POLICY_CKPT}"
  "runtime.interactive=${INTERACTIVE}"
  "runtime.viewer_enabled=${VIEWER_ENABLED}"
  "runtime.image_obs_fps=${IMAGE_OBS_FPS}"
  "runtime.policy_fps=${POLICY_FPS}"
  "runtime.spacemouse_fps=${SPACEMOUSE_FPS}"
  "runtime.eval_episode_max_steps=${EPISODE_MAX_STEPS}"
  "intervention.enabled=${INTERVENTION_ENABLED}"
  "algorithm.flow.device=${POLICY_DEVICE}"
  "algorithm.flow.inference_device=${POLICY_DEVICE}"
  "algorithm.flow.action_horizon=${ACTION_HORIZON}"
  "algorithm.flow.execute_horizon=${EXECUTE_HORIZON}"
  "algorithm.flow.n_ode_steps=${N_ODE_STEPS}"
  "algorithm.discriminator.checkpoint=${NNPU_CKPT}"
  "algorithm.discriminator.inference.device=${DISC_DEVICE}"
  "algorithm.discriminator.learner_device=${DISC_DEVICE}"
  "algorithm.discriminator.inference.fps=${DISC_INFERENCE_FPS}"
  "algorithm.discriminator.inference.intervene_env=${DISC_INTERVENE_ENV}"
  "logging.use_tensorboard=false"
  "logging.use_wandb=false"
  "+offline_collect.num_episodes=${NUM_TRAJECTORIES}"
  "+offline_collect.output_dir=${OUTPUT_DIR}"
  "+offline_collect.output_file=${OUTPUT_FILE}"
  "+offline_collect.episode_max_steps=${EPISODE_MAX_STEPS}"
  "+offline_collect.deterministic=${DETERMINISTIC}"
  "+offline_collect.save_every_episode=${SAVE_EVERY_EPISODE}"
  "+offline_collect.save_interval_episodes=${SAVE_INTERVAL_EPISODES}"
)
HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

echo "[collect_data] task=${TASK} episodes=${NUM_TRAJECTORIES}"
echo "[collect_data] policy_device=${POLICY_DEVICE} disc_device=${DISC_DEVICE}"
echo "[collect_data] output=${OUTPUT_DIR}/${OUTPUT_FILE}"
echo "[collect_data] policy_ckpt=${POLICY_CKPT}"
echo "[collect_data] nnpu_ckpt=${NNPU_CKPT}"

"${PY}" -m robosuite.pipeline.offline.utils.collect_data "${HYDRA_OVERRIDES[@]}" "$@"
