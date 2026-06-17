#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

# ---------------------------------------------------------
ENV_NAME="Stack"
TASK_NAME="PandaStack"
CHECKPOINT="outputs/hil_serl/hil_serl_Stack_2026-04-25_21-22-59/checkpoints/step_00020000_updates_00011830_ep_00108.pt"
EPISODES=50
EVAL_EPISODE_MAX_STEPS=400
VIDEO_OUTPUT="${VIDEO_OUTPUT:-false}"
# ---------------------------------------------------------

VIDEO_CAMERA="${VIDEO_CAMERA:-agentview}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_IMAGE_SIZE="${VIDEO_IMAGE_SIZE:-512}"

INTERACTIVE="${INTERACTIVE:-false}"
EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-true}"
DEVICE="${DEVICE:-}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-}"
CAMERA_NAMES="${CAMERA_NAMES:-}"
IMAGE_SIZE="${IMAGE_SIZE:-}"

EXTRA_ARGS=("$@")

export ROOT_DIR
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

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
  echo "  CHECKPOINT=/abs/path/to/hil_serl_ckpt.pt [ENV_NAME=Stack] [TASK_NAME=PandaStack] [EPISODES=20] bash robosuite/pipeline/scripts/eval_hil_serl.sh" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint file does not exist: ${CHECKPOINT}" >&2
  exit 1
fi

echo "[eval] checkpoint=${CHECKPOINT}"
echo "[eval] env=${ENV_NAME} task=${TASK_NAME} episodes=${EPISODES} deterministic=${EVAL_DETERMINISTIC} video_output=${VIDEO_OUTPUT}"
echo "[eval] results_dir=${ROOT_DIR}/outputs/hil_serl/eval"

PY_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --env-name "${ENV_NAME}"
  --task-name "${TASK_NAME}"
  --episodes "${EPISODES}"
  --episode-max-steps "${EVAL_EPISODE_MAX_STEPS}"
  --output-root "./outputs/hil_serl/eval"
  --video-output "${VIDEO_OUTPUT}"
  --video-camera "${VIDEO_CAMERA}"
  --video-fps "${VIDEO_FPS}"
  --video-height "${VIDEO_IMAGE_SIZE}"
  --video-width "${VIDEO_IMAGE_SIZE}"
)

if [[ "${INTERACTIVE}" == "true" ]]; then
  PY_ARGS+=(--interactive)
fi

if [[ "${EVAL_DETERMINISTIC}" == "true" ]]; then
  PY_ARGS+=(--deterministic)
else
  PY_ARGS+=(--no-deterministic)
fi

if [[ -n "${DEVICE}" ]]; then
  PY_ARGS+=(--device "${DEVICE}")
fi

if [[ -n "${INFERENCE_DEVICE}" ]]; then
  PY_ARGS+=(--inference-device "${INFERENCE_DEVICE}")
fi

if [[ -n "${CAMERA_NAMES}" ]]; then
  PY_ARGS+=(--camera-names "${CAMERA_NAMES}")
fi

if [[ -n "${IMAGE_SIZE}" ]]; then
  PY_ARGS+=(--image-size "${IMAGE_SIZE}")
fi

"${PYTHON_BIN}" -m robosuite.pipeline.eval_hil_serl \
  "${PY_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

# Examples:
#   CHECKPOINT=/abs/path/to/latest.pt bash robosuite/pipeline/scripts/eval_hil_serl.sh
#   CHECKPOINT=/abs/path/to/latest.pt ENV_NAME=Stack TASK_NAME=PandaStack EPISODES=50 \
#     bash robosuite/pipeline/scripts/eval_hil_serl.sh
#   CHECKPOINT=/abs/path/to/latest.pt VIDEO_OUTPUT=true bash robosuite/pipeline/scripts/eval_hil_serl.sh
