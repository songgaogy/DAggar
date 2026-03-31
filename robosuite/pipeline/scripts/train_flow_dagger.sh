#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"

ENVIRONMENT="Stack"
DEMO_TASK_NAME="PandaStack"
NUM_TRAJECTORIES=20
FLOW_DAGGER_MODE="train"    # eval or train

INTERACTIVE="${INTERACTIVE:-true}"
VIEWER_ENABLED="${VIEWER_ENABLED:-true}"
IMAGE_OBS_FPS="${IMAGE_OBS_FPS:-20}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-0}"
LEARNER_DEVICE="${LEARNER_DEVICE:-cuda:0}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda:1}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-4}"
N_ODE_STEPS=10
EVAL_EPISODE_MAX_STEPS="${EVAL_EPISODE_MAX_STEPS:-300}"
INIT_CHECKPOINT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt"
FPS_LOG_INTERVAL="${FPS_LOG_INTERVAL:-3.0}"
UNTHROTTLED="${UNTHROTTLED:-false}"

case "${FLOW_DAGGER_MODE}" in
  eval)
    ONLINE_UPDATES="${ONLINE_UPDATES:-false}"
    INTERVENTION_ENABLED="${INTERVENTION_ENABLED:-false}"
    ASYNC_UPDATES="${ASYNC_UPDATES:-false}"
    VISUALIZE_GRIPPER_MARKERS="${VISUALIZE_GRIPPER_MARKERS:-false}"
    LOGGING_USE_WANDB="${LOGGING_USE_WANDB:-false}"
    VIEWER_ASYNC="${VIEWER_ASYNC:-false}"
    VIEWER_BACKEND="${VIEWER_BACKEND:-mjviewer}"
    RENDER_FPS="${RENDER_FPS:-20}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
    BUFFER_SAVE_INTERVAL="${BUFFER_SAVE_INTERVAL:-1000}"
    CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5000}"
    LOG_INTERVAL="${LOG_INTERVAL:-50}"
    PUBLISH_INTERVAL="${PUBLISH_INTERVAL:-100}"
    SEED_VALUE="${SEED:-null}"
    EPISODE_PAUSE_SEC="${EPISODE_PAUSE_SEC:-2}"
    ;;
  train)
    ONLINE_UPDATES="${ONLINE_UPDATES:-true}"
    INTERVENTION_ENABLED="${INTERVENTION_ENABLED:-true}"
    ASYNC_UPDATES="${ASYNC_UPDATES:-true}"
    VISUALIZE_GRIPPER_MARKERS="${VISUALIZE_GRIPPER_MARKERS:-true}"
    LOGGING_USE_WANDB="${LOGGING_USE_WANDB:-false}"
    VIEWER_ASYNC="${VIEWER_ASYNC:-true}"
    VIEWER_BACKEND="${VIEWER_BACKEND:-mjviewer}"
    RENDER_FPS="${RENDER_FPS:-10}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
    BUFFER_SAVE_INTERVAL="${BUFFER_SAVE_INTERVAL:-5000}"
    CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
    LOG_INTERVAL="${LOG_INTERVAL:-100}"
    PUBLISH_INTERVAL="${PUBLISH_INTERVAL:-300}"
    SEED_VALUE="${SEED:-42}"
    EPISODE_PAUSE_SEC="${EPISODE_PAUSE_SEC:-0}"
    ;;
  *)
    echo "[ERROR] Unsupported FLOW_DAGGER_MODE='${FLOW_DAGGER_MODE}'. Use 'eval' or 'train'." >&2
    exit 1
    ;;
esac

LOAD="${LOAD:-null}"
RESUME="${RESUME:-false}"
CHECKPOINT="${CHECKPOINT:-null}"
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
  seed="${SEED_VALUE}" \
  env.environment="${ENVIRONMENT}" \
  env.renderer="mjviewer" \
  data.task_name="${DEMO_TASK_NAME}" \
  data.num_trajectories="${NUM_TRAJECTORIES}" \
  runtime.interactive="${INTERACTIVE}" \
  runtime.viewer_enabled="${VIEWER_ENABLED}" \
  runtime.visualize_gripper_markers="${VISUALIZE_GRIPPER_MARKERS}" \
  runtime.image_obs_fps="${IMAGE_OBS_FPS}" \
  runtime.render_fps="${RENDER_FPS}" \
  runtime.viewer_async="${VIEWER_ASYNC}" \
  runtime.viewer_backend="${VIEWER_BACKEND}" \
  runtime.fps_log_interval="${FPS_LOG_INTERVAL}" \
  runtime.buffer_save_interval="${BUFFER_SAVE_INTERVAL}" \
  runtime.unthrottled="${UNTHROTTLED}" \
  runtime.async_updates="${ASYNC_UPDATES}" \
  runtime.online_updates_enabled="${ONLINE_UPDATES}" \
  runtime.eval_episode_max_steps="${EVAL_EPISODE_MAX_STEPS}" \
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
  algorithm.flow.n_ode_steps="${N_ODE_STEPS}" \
  algorithm.trainer.batch_size="${TRAIN_BATCH_SIZE}" \
  algorithm.trainer.steps_per_update="${PUBLISH_INTERVAL}" \
  logging.use_wandb="${LOGGING_USE_WANDB}" \
  logging.log_interval="${LOG_INTERVAL}" \
  logging.checkpoint_interval="${CHECKPOINT_INTERVAL}" \
  "${EXTRA_ARGS[@]}"

# Evaluate with window rendering:
# FLOW_DAGGER_MODE=eval bash robosuite/pipeline/scripts/train_flow_dagger.sh
#
# Online training / intervention:
# FLOW_DAGGER_MODE=train bash robosuite/pipeline/scripts/train_flow_dagger.sh
