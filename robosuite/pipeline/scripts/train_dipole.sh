#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

ENVIRONMENT="${ENVIRONMENT:-PickPlaceBread}"
DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-20}"
DIPOLE_MODE="${DIPOLE_MODE:-train}"   # eval or train

INTERACTIVE="${INTERACTIVE:-true}"
VIEWER_ENABLED="${VIEWER_ENABLED:-true}"
IMAGE_OBS_FPS="${IMAGE_OBS_FPS:-20}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-0}"
LEARNER_DEVICE="${LEARNER_DEVICE:-cuda:0}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda:1}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-4}"
N_ODE_STEPS="${N_ODE_STEPS:-10}"
EVAL_EPISODE_MAX_STEPS="${EVAL_EPISODE_MAX_STEPS:-300}"
FPS_LOG_INTERVAL="${FPS_LOG_INTERVAL:-3.0}"
DISC_INFERENCE_FPS="${DISC_INFERENCE_FPS:--1}"
DISC_INTERVENE_ENV="${DISC_INTERVENE_ENV:-true}"
UNTHROTTLED="${UNTHROTTLED:-false}"

# ----------------------------------------------------------------------------------------------
# REQUIRED: flow-policy base checkpoint and task-calibrated nnPU head.
INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
NNPU_CKPT="${NNPU_CKPT:-}"
NNPU_ENCODER_CKPT="${NNPU_ENCODER_CKPT:-null}"

# DIPOLE hyperparameters.
BETA="${BETA:-1.0}"
K="${K:-0.0}"
OMEGA="${OMEGA:-0.2}"
OMEGA_RUNTIME_OVERRIDE="${OMEGA_RUNTIME_OVERRIDE:-}"
G_CLIP="${G_CLIP:-10.0}"
POLARITY_INIT="${POLARITY_INIT:-zero_pos}"
POLARITY_INIT_SCALE="${POLARITY_INIT_SCALE:-1.0e-3}"

# Optional: camera_to_view JSON object, e.g. '{robot0_robotview: agentview}'.
# Leave empty to use identity (policy camera_name == encoder view_name).
NNPU_CAMERA_TO_VIEW="${NNPU_CAMERA_TO_VIEW:-{}}"
# ----------------------------------------------------------------------------------------------


case "${DIPOLE_MODE}" in
  eval)
    ONLINE_UPDATES="${ONLINE_UPDATES:-false}"
    INTERVENTION_ENABLED="${INTERVENTION_ENABLED:-false}"
    ASYNC_UPDATES="${ASYNC_UPDATES:-false}"
    VISUALIZE_GRIPPER_MARKERS="${VISUALIZE_GRIPPER_MARKERS:-false}"
    LOGGING_USE_TENSORBOARD="${LOGGING_USE_TENSORBOARD:-false}"
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
    LOGGING_USE_TENSORBOARD="${LOGGING_USE_TENSORBOARD:-true}"
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
  echo "[ERROR] intervention.enabled=true currently requires DISPLAY (robosuite input devices need pynput)." >&2
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
      echo "[ERROR] LOAD points to a directory but no latest checkpoint was found under ${LOAD}" >&2
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
if [[ -z "${NNPU_CKPT}" || ! -f "${NNPU_CKPT}" ]]; then
  echo "[ERROR] NNPU_CKPT must point to a task-calibrated pu_bce_head.pth: ${NNPU_CKPT:-<unset>}" >&2
  exit 1
fi

EFFECTIVE_OMEGA="${OMEGA_RUNTIME_OVERRIDE:-${OMEGA}}"

TRAIN_DIPOLE_CMD=(
  "${PYTHON_BIN}" -m robosuite.pipeline.train_dipole
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
  algorithm.dipole.beta="${BETA}" \
  algorithm.dipole.k="${K}" \
  algorithm.dipole.guidance_omega="${EFFECTIVE_OMEGA}" \
  algorithm.dipole.g_clip="${G_CLIP}" \
  algorithm.dipole.polarity_embedding_init="${POLARITY_INIT}" \
  algorithm.dipole.polarity_embedding_init_scale="${POLARITY_INIT_SCALE}" \
  algorithm.discriminator.checkpoint="${NNPU_CKPT}" \
  algorithm.discriminator.encoder_ckpt="${NNPU_ENCODER_CKPT}" \
  algorithm.discriminator.camera_to_view="${NNPU_CAMERA_TO_VIEW}" \
  algorithm.discriminator.inference.fps="${DISC_INFERENCE_FPS}" \
  algorithm.discriminator.inference.intervene_env="${DISC_INTERVENE_ENV}" \
  logging.use_tensorboard="${LOGGING_USE_TENSORBOARD}" \
  logging.use_wandb="${LOGGING_USE_WANDB}" \
  logging.log_interval="${LOG_INTERVAL}" \
  logging.checkpoint_interval="${CHECKPOINT_INTERVAL}" \
  hydra/hydra_logging=none \
  hydra/job_logging=none \
  "${EXTRA_ARGS[@]}"
)

"${TRAIN_DIPOLE_CMD[@]}"

# Train with intervention + online updates:
#   INIT_CHECKPOINT=/abs/flow_dagger_latest.pt \
#   NNPU_CKPT=checkpoints/dyn_disc/.../checkpoints/pu_bce_head.pth \
#   bash robosuite/pipeline/scripts/train_dipole.sh
#
# Sweep omega at eval time without retraining:
#   DIPOLE_MODE=eval OMEGA_RUNTIME_OVERRIDE=4 \
#   INIT_CHECKPOINT=... NNPU_CKPT=... \
#   bash robosuite/pipeline/scripts/train_dipole.sh
