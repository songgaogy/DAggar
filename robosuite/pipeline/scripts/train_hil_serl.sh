#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"

ENVIRONMENT="Lift"
DEMO_TASK_NAME="PandaLift"  # demo data
NUM_TRAJECTORIES=20
INTERACTIVE=true
VIEWER_ENABLED=true
VIEWER_BACKEND=auto
VIEWER_STARTUP_DELAY="1"
VIEWER_RESET_WARMUP_FRAMES=2
VISUALIZE_GRIPPER_MARKERS=true
IMAGE_OBS_FPS="10"
INTERVENTION_ENABLED=true
ASYNC_UPDATES=true
LEARNER_DEVICE="cuda:0"
INFERENCE_DEVICE="cuda:1"
LOGGING_USE_WANDB=true
RESUME="${RESUME:-true}"
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
python -m robosuite.pipeline.train_hil_serl \
  env.environment="${ENVIRONMENT}" \
  data.task_name="${DEMO_TASK_NAME}" \
  data.num_trajectories="${NUM_TRAJECTORIES}" \
  runtime.interactive="${INTERACTIVE}" \
  runtime.viewer_enabled="${VIEWER_ENABLED}" \
  runtime.viewer_backend="${VIEWER_BACKEND}" \
  runtime.viewer_startup_delay="${VIEWER_STARTUP_DELAY}" \
  runtime.viewer_reset_warmup_frames="${VIEWER_RESET_WARMUP_FRAMES}" \
  runtime.visualize_gripper_markers="${VISUALIZE_GRIPPER_MARKERS}" \
  runtime.image_obs_fps="${IMAGE_OBS_FPS}" \
  runtime.async_updates="${ASYNC_UPDATES}" \
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
# 6. Force mjviewer back on for experiments:
#    VIEWER_BACKEND=mjviewer VIEWER_STARTUP_DELAY=0.2 \
#    bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 7. Increase OpenCV reset warmup if preview still shows snow after reset:
#    VIEWER_RESET_WARMUP_FRAMES=4 bash robosuite/pipeline/scripts/train_hil_serl.sh
#
# 8. Disable gripper visualization markers if needed:
#    VISUALIZE_GRIPPER_MARKERS=false bash robosuite/pipeline/scripts/train_hil_serl.sh
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
