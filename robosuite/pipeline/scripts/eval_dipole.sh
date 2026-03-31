#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"

ENV_NAME="Stack"
TASK_NAME="PandaStack"
CHECKPOINT="/home/dodo/Documents/DAggar/robosuite/outputs/dipole/dipole_Stack_2026-03-31_21-17-49/checkpoints/step_00010000_updates_00011100_ep_00037.pt"
EPISODES=20
EVAL_EPISODE_MAX_STEPS=500
VIDEO_OUTPUT="${VIDEO_OUTPUT:-true}"
VIDEO_IMAGE_SIZE="${VIDEO_IMAGE_SIZE:-512}"

INTERACTIVE="${INTERACTIVE:-false}"
VIDEO_CAMERA="${VIDEO_CAMERA:-agentview}"
VIDEO_FPS="${VIDEO_FPS:-20}"
EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-false}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

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
  echo "  CHECKPOINT=/abs/path/to/dipole_ckpt.pt [ENV_NAME=Stack] [TASK_NAME=PandaStack] [EPISODES=5] bash robosuite/pipeline/scripts/eval_dipole.sh" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint file does not exist: ${CHECKPOINT}" >&2
  exit 1
fi

echo "[eval] checkpoint=${CHECKPOINT}"
echo "[eval] env=${ENV_NAME} task=${TASK_NAME} episodes=${EPISODES} video_output=${VIDEO_OUTPUT}"
echo "[eval] results_dir=${ROOT_DIR}/outputs/dipole/eval"

PY_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --env-name "${ENV_NAME}"
  --task-name "${TASK_NAME}"
  --episodes "${EPISODES}"
  --episode-max-steps "${EVAL_EPISODE_MAX_STEPS}"
  --output-root "./outputs/dipole/eval"
  --video-output "${VIDEO_OUTPUT}"
  --video-camera "${VIDEO_CAMERA}"
  --video-fps "${VIDEO_FPS}"
  --video-height "${VIDEO_IMAGE_SIZE}"
  --video-width "${VIDEO_IMAGE_SIZE}"
)

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  PY_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

if [[ "${INTERACTIVE}" == "true" ]]; then
  PY_ARGS+=(--interactive)
fi

if [[ "${EVAL_DETERMINISTIC}" == "true" ]]; then
  PY_ARGS+=(--deterministic)
fi

python -m robosuite.pipeline.eval_dipole \
  "${PY_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

# Examples:
#   CHECKPOINT=/abs/path/to/latest.pt bash robosuite/pipeline/scripts/eval_dipole.sh
#   CHECKPOINT=/abs/path/to/latest.pt ENV_NAME=Stack TASK_NAME=PandaStack EPISODES=10 \
#     bash robosuite/pipeline/scripts/eval_dipole.sh
#   CHECKPOINT=/abs/path/to/latest.pt VIDEO_OUTPUT=true bash robosuite/pipeline/scripts/eval_dipole.sh
