#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"

ENV_NAME="PickPlaceBread"
TASK_NAME="${TASK_NAME:-$ENV_NAME}"
CHECKPOINT="outputs/flow-DAgger/flow_dagger_PickPlaceBread_2026-04-26_20-43-06/checkpoints/step_00010000_updates_00009711_ep_00040.pt"
EPISODES=50
EVAL_EPISODE_MAX_STEPS=700
VIDEO_OUTPUT="true"
VIDEO_IMAGE_SIZE=512

INTERACTIVE="false"
VIEWER_ENABLED="false"
VISUALIZE_GRIPPER_MARKERS="false"
IMAGE_OBS_FPS="20"
RENDER_FPS="20"
FPS_LOG_INTERVAL="3.0"
UNTHROTTLED="false"
NUM_TRAJECTORIES="20"
MAX_STEPS="$((EPISODES * EVAL_EPISODE_MAX_STEPS))"
EPISODE_PAUSE_SEC="0"
EVAL_DETERMINISTIC="false"
VIDEO_OUTPUT="${VIDEO_OUTPUT:-false}"
SEED_VALUE="null"

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
  echo "Use a GUI terminal, or run INTERACTIVE=false for headless evaluation." >&2
  exit 1
fi

cd "${ROOT_DIR}"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] Missing CHECKPOINT." >&2
  echo "Usage:" >&2
  echo "  CHECKPOINT=/abs/path/to/ckpt.pt [ENV_NAME=PickPlaceBread] [TASK_NAME=PickPlaceBread] [EPISODES=5] bash robosuite/pipeline/scripts/eval_flow_dagger.sh" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint file does not exist: ${CHECKPOINT}" >&2
  exit 1
fi

echo "[eval] checkpoint=${CHECKPOINT}"
echo "[eval] env=${ENV_NAME} task=${TASK_NAME} episodes=${EPISODES} video_output=${VIDEO_OUTPUT}"
echo "[eval] results_dir=${ROOT_DIR}/outputs/flow-DAgger/eval"

PY_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --env-name "${ENV_NAME}"
  --task-name "${TASK_NAME}"
  --episodes "${EPISODES}"
  --episode-max-steps "${EVAL_EPISODE_MAX_STEPS}"
  --output-root "./outputs/flow-DAgger/eval"
  --video-output "${VIDEO_OUTPUT}"
  --video-height "${VIDEO_IMAGE_SIZE}"
  --video-width "${VIDEO_IMAGE_SIZE}"
)

if [[ "${INTERACTIVE}" == "true" ]]; then
  PY_ARGS+=(--interactive)
fi

if [[ "${EVAL_DETERMINISTIC}" == "true" ]]; then
  PY_ARGS+=(--deterministic)
fi

python -m robosuite.pipeline.eval_flow_dagger \
  "${PY_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

# Examples:
#   CHECKPOINT=/abs/path/to/latest.pt bash robosuite/pipeline/scripts/eval_flow_dagger.sh
#   CHECKPOINT=/abs/path/to/latest.pt ENV_NAME=PickPlaceBread TASK_NAME=PickPlaceBread EPISODES=10 \
#     bash robosuite/pipeline/scripts/eval_flow_dagger.sh
#   VIDEO_OUTPUT=true bash robosuite/pipeline/scripts/eval_flow_dagger.sh
