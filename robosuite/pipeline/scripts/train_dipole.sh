#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"

ENVIRONMENT="${ENVIRONMENT:-Lift}"
DEMO_TASK_NAME="${DEMO_TASK_NAME:-}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-20}"
DIPOLE_MODE="${DIPOLE_MODE:-train}"    # eval or train

INTERACTIVE="${INTERACTIVE:-true}"
VIEWER_ENABLED="${VIEWER_ENABLED:-true}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-0}"
LEARNER_DEVICE="${LEARNER_DEVICE:-cuda:0}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda:1}"
DISCRIMINATOR_DEVICE="${DISCRIMINATOR_DEVICE:-cuda:0}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-4}"
N_ODE_STEPS="${N_ODE_STEPS:-10}"
BETA="${BETA:-2.0}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
LPB_CHECKPOINT="${LPB_CHECKPOINT:-/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/lpb_new/lpb_new_20260324_232921/lpb_new_20260324_232921_ep0050.pt}"
FPS_LOG_INTERVAL="${FPS_LOG_INTERVAL:-3.0}"
UNTHROTTLED="${UNTHROTTLED:-false}"

case "${DIPOLE_MODE}" in
  eval)
    ONLINE_UPDATES="${ONLINE_UPDATES:-false}"
    INTERVENTION_ENABLED="${INTERVENTION_ENABLED:-false}"
    ASYNC_UPDATES="${ASYNC_UPDATES:-false}"
    VISUALIZE_GRIPPER_MARKERS="${VISUALIZE_GRIPPER_MARKERS:-false}"
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
    echo "[ERROR] Unsupported DIPOLE_MODE='${DIPOLE_MODE}'. Use 'eval' or 'train'." >&2
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
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-songgao-personal}"
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
  exit 1
fi

if [[ "${INTERVENTION_ENABLED}" == "true" && -z "${DISPLAY:-}" ]]; then
  echo "[ERROR] intervention.enabled=true currently requires DISPLAY because robosuite input devices import pynput." >&2
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

DEMO_TASK_ARGS=()
if [[ -n "${DEMO_TASK_NAME}" && "${DEMO_TASK_NAME}" != "null" ]]; then
  DEMO_TASK_ARGS=("data.task_name=${DEMO_TASK_NAME}")
fi

python -m robosuite.pipeline.train_dipole \
  seed="${SEED_VALUE}" \
  env.environment="${ENVIRONMENT}" \
  env.renderer="mjviewer" \
  data.num_trajectories="${NUM_TRAJECTORIES}" \
  "${DEMO_TASK_ARGS[@]}" \
  runtime.interactive="${INTERACTIVE}" \
  runtime.viewer_enabled="${VIEWER_ENABLED}" \
  runtime.visualize_gripper_markers="${VISUALIZE_GRIPPER_MARKERS}" \
  runtime.render_fps="${RENDER_FPS}" \
  runtime.viewer_async="${VIEWER_ASYNC}" \
  runtime.viewer_backend="${VIEWER_BACKEND}" \
  runtime.fps_log_interval="${FPS_LOG_INTERVAL}" \
  runtime.buffer_save_interval="${BUFFER_SAVE_INTERVAL}" \
  runtime.unthrottled="${UNTHROTTLED}" \
  runtime.async_updates="${ASYNC_UPDATES}" \
  runtime.online_updates_enabled="${ONLINE_UPDATES}" \
  runtime.episode_pause_sec="${EPISODE_PAUSE_SEC}" \
  intervention.enabled="${INTERVENTION_ENABLED}" \
  runtime.resume="${RESUME}" \
  runtime.checkpoint="${CHECKPOINT}" \
  runtime.init_checkpoint="${INIT_CHECKPOINT}" \
  discriminator.model.lpb_ckpt="${LPB_CHECKPOINT}" \
  discriminator.policy.ckpt="${INIT_CHECKPOINT}" \
  discriminator.policy.device="${DISCRIMINATOR_DEVICE}" \
  discriminator.feature.device="${DISCRIMINATOR_DEVICE}" \
  discriminator.detector.device="${DISCRIMINATOR_DEVICE}" \
  algorithm.trainer.pretrain_steps="${PRETRAIN_STEPS}" \
  algorithm.dipole.device="${LEARNER_DEVICE}" \
  algorithm.dipole.inference_device="${INFERENCE_DEVICE}" \
  algorithm.dipole.action_horizon="${ACTION_HORIZON}" \
  algorithm.dipole.execute_horizon="${EXECUTE_HORIZON}" \
  algorithm.dipole.n_ode_steps="${N_ODE_STEPS}" \
  algorithm.dipole.beta="${BETA}" \
  algorithm.dipole.guidance_scale="${GUIDANCE_SCALE}" \
  algorithm.trainer.batch_size="${TRAIN_BATCH_SIZE}" \
  algorithm.trainer.steps_per_update="${PUBLISH_INTERVAL}" \
  logging.log_interval="${LOG_INTERVAL}" \
  logging.checkpoint_interval="${CHECKPOINT_INTERVAL}" \
  "${EXTRA_ARGS[@]}"

# Example:
# DIPOLE_MODE=train bash robosuite/pipeline/scripts/train_dipole.sh
