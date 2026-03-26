#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"

ENVIRONMENT="Lift"
DEMO_TASK_NAME="PandaLift"  # demo data
NUM_TRAJECTORIES=20
INTERACTIVE=true
VIEWER_ENABLED=true
VISUALIZE_GRIPPER_MARKERS=true
IMAGE_OBS_FPS="10"
INTERVENTION_ENABLED=true
ASYNC_UPDATES=true
LEARNER_DEVICE="cuda:0"
INFERENCE_DEVICE="cuda:1"
LOGGING_USE_WANDB=true

# previous log & ckpt path & data
LOAD="null"

# previous ckpt only, not useful
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

python -m robosuite.pipeline.train_hil_serl \
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
  algorithm.sac.device="${LEARNER_DEVICE}" \
  algorithm.sac.inference_device="${INFERENCE_DEVICE}" \
  logging.use_wandb="${LOGGING_USE_WANDB}" \
  "${EXTRA_ARGS[@]}"

# Examples:
# 1. Run with defaults:
#    bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 2. Override a few common fields:
#    ENVIRONMENT=PickPlaceCan DEMO_TASK_NAME=PandaPickPlaceCan NUM_TRAJECTORIES=50 \
#    bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 3. Disable intervention and run autonomous training:
#    INTERVENTION_ENABLED=false bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 4. Run fully headless:
#    INTERACTIVE=false VIEWER_ENABLED=false INTERVENTION_ENABLED=false \
#    bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 5. Disable async learner to isolate timing-sensitive crashes:
#    ASYNC_UPDATES=false bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 6. Load from a previous run directory or a checkpoint file:
#    LOAD=outputs/hil_serl/hil_serl_Lift_2026-03-25_21-14-34 \
#    bash robosuite/pipeline/scripts/train_hil_serl.sh
#    LOAD=/abs/path/to/checkpoints/latest.pt bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 7. Disable gripper visualization markers if needed:
#    VISUALIZE_GRIPPER_MARKERS=false bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 8. Change the episode pause after each finished task:
#    EPISODE_PAUSE_SEC=0.2 bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 9. Run with slower image observations and auto dual-GPU split:
#    IMAGE_OBS_FPS=10 LEARNER_DEVICE=cuda:0 INFERENCE_DEVICE=auto \
#    bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 10. Resume from a specific checkpoint:
#    CHECKPOINT=/abs/path/latest.pt RESUME=true bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 11. Pass extra Hydra overrides directly when you really need them:
#    bash robosuite/pipeline/scripts/train_hil_serl.sh \
#      algorithm.trainer.batch_size=128 \
#      algorithm.trainer.cta_ratio=4 \
#      seed=0
