#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"

ENVIRONMENT="PickPlaceBread"
DEMO_TASK_NAME="PickPlaceBread"
NUM_TRAJECTORIES=20
INTERACTIVE="${INTERACTIVE:-true}"
VIEWER_ENABLED="${VIEWER_ENABLED:-true}"
VISUALIZE_GRIPPER_MARKERS="${VISUALIZE_GRIPPER_MARKERS:-true}"
IMAGE_OBS_FPS="${IMAGE_OBS_FPS:-10}"
INTERVENTION_ENABLED="${INTERVENTION_ENABLED:-true}"
ASYNC_UPDATES="${ASYNC_UPDATES:-true}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-0}"
LEARNER_DEVICE="${LEARNER_DEVICE:-cuda:0}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-auto}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-4}"
LOGGING_USE_WANDB="true"
INIT_CHECKPOINT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt"

LOAD="${LOAD:-null}"
RESUME="${RESUME:-false}"
CHECKPOINT="${CHECKPOINT:-null}"
EPISODE_PAUSE_SEC="${EPISODE_PAUSE_SEC:-2}"

EXTRA_ARGS=("$@")

export ROOT_DIR
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

if [[ "${INTERACTIVE}" == "true" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

if [[ "${INTERACTIVE}" == "true" && -z "${DISPLAY:-}" ]]; then
  echo "[ERROR] INTERACTIVE=true requires a desktop X session, but DISPLAY is empty." >&2
  echo "Use the VSCode debugger / a GUI terminal, or run INTERACTIVE=false for headless mode." >&2
  exit 1
fi

if [[ "${INTERVENTION_ENABLED}" == "true" && -z "${DISPLAY:-}" ]]; then
  echo "[ERROR] intervention.enabled=true currently requires DISPLAY because robosuite input devices import pynput." >&2
  echo "Use a GUI terminal, or run INTERVENTION_ENABLED=false for headless training." >&2
  exit 1
fi

cd "${ROOT_DIR}"

if [[ -n "${LOAD}" && "${LOAD}" != "null" ]]; then
  if [[ -d "${LOAD}" ]]; then
    if [[ -f "${LOAD}/checkpoints/latest.pt" ]]; then
      CHECKPOINT="${LOAD}/checkpoints/latest.pt"
    elif [[ -f "${LOAD}/latest.pt" ]]; then
      CHECKPOINT="${LOAD}/latest.pt"
    else
      echo "[ERROR] LOAD points to a directory, but no latest checkpoint was found under ${LOAD}" >&2
      exit 1
    fi
  else
    CHECKPOINT="${LOAD}"
  fi
  RESUME=true
fi

if [[ "${CHECKPOINT}" != "null" && ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint file does not exist: ${CHECKPOINT}" >&2
  exit 1
fi

python -m robosuite.pipeline.train_flow_dagger \
  env.environment="${ENVIRONMENT}" \
  env.renderer="mjviewer" \
  data.task_name="${DEMO_TASK_NAME}" \
  data.num_trajectories="${NUM_TRAJECTORIES}" \
  runtime.interactive="${INTERACTIVE}" \
  runtime.viewer_enabled="${VIEWER_ENABLED}" \
  runtime.visualize_gripper_markers="${VISUALIZE_GRIPPER_MARKERS}" \
  runtime.image_obs_fps="${IMAGE_OBS_FPS}" \
  runtime.async_updates="${ASYNC_UPDATES}" \
  runtime.episode_pause_sec="${EPISODE_PAUSE_SEC}" \
  intervention.enabled="${INTERVENTION_ENABLED}" \
  runtime.resume="${RESUME}" \
  runtime.checkpoint="${CHECKPOINT}" \
  runtime.init_checkpoint="${INIT_CHECKPOINT}" \
  algorithm.trainer.pretrain_steps="${PRETRAIN_STEPS}" \
  algorithm.flow.device="${LEARNER_DEVICE}" \
  algorithm.flow.inference_device="${INFERENCE_DEVICE}" \
  algorithm.flow.action_horizon="${ACTION_HORIZON}" \
  algorithm.flow.execute_horizon="${EXECUTE_HORIZON}" \
  logging.use_wandb="${LOGGING_USE_WANDB}" \
  "${EXTRA_ARGS[@]}"
