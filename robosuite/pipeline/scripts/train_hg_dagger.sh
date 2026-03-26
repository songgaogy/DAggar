#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"

ENVIRONMENT="Lift"
DEMO_TASK_NAME="PandaLift"
NUM_TRAJECTORIES=20
INTERACTIVE=true
VIEWER_ENABLED=true
VISUALIZE_GRIPPER_MARKERS=true
IMAGE_OBS_FPS="10"
INTERVENTION_ENABLED=true
ASYNC_UPDATES=true
PRETRAIN_STEPS=0
LEARNER_DEVICE="cuda:0"
INFERENCE_DEVICE="cuda:1"
LOGGING_USE_WANDB=true

LOAD="${LOAD:-null}"
RESUME="${RESUME:-false}"
CHECKPOINT="${CHECKPOINT:-null}"
EPISODE_PAUSE_SEC=2

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

python -m robosuite.pipeline.train_hg_dagger \
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
  algorithm.trainer.pretrain_steps="${PRETRAIN_STEPS}" \
  algorithm.bc.device="${LEARNER_DEVICE}" \
  algorithm.bc.inference_device="${INFERENCE_DEVICE}" \
  logging.use_wandb="${LOGGING_USE_WANDB}" \
  "${EXTRA_ARGS[@]}"

# Examples:
# 1. Run with defaults:
#    bash robosuite/pipeline/scripts/train_hg_dagger.sh
#
# 2. Override task and demo count:
#    ENVIRONMENT=PickPlaceCan DEMO_TASK_NAME=PandaPickPlaceCan NUM_TRAJECTORIES=50 \
#    bash robosuite/pipeline/scripts/train_hg_dagger.sh
#
# 3. Disable intervention and run headless BC fine-tuning:
#    INTERACTIVE=false VIEWER_ENABLED=false INTERVENTION_ENABLED=false \
#    bash robosuite/pipeline/scripts/train_hg_dagger.sh
#
# 4. Change BC pretraining length:
#    PRETRAIN_STEPS=5000 bash robosuite/pipeline/scripts/train_hg_dagger.sh
#
# 5. Resume from a previous run directory or checkpoint:
#    LOAD=outputs/HG-DAgger/hg_dagger_Lift_2026-03-26_12-00-00 \
#    bash robosuite/pipeline/scripts/train_hg_dagger.sh
#    LOAD=/abs/path/to/checkpoints/latest.pt bash robosuite/pipeline/scripts/train_hg_dagger.sh
#
# 6. Pass extra Hydra overrides directly:
#    bash robosuite/pipeline/scripts/train_hg_dagger.sh \
#      algorithm.trainer.batch_size=128 \
#      algorithm.trainer.updates_per_step=2 \
#      seed=0
