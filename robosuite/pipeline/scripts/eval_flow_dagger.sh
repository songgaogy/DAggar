#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
export CUDA_VISIBLE_DEVICES=1

# -------------------------------------------------------------------------------------------------
ENV_NAME="NutAssemblySquare"
CHECKPOINT="outputs/flow-DAgger/flow_dagger_NutAssemblySquare_2026-06-16_21-56-22_30pretrain/checkpoints/step_00015000_updates_00009964_ep_00039.pt"
EPISODES=50
EVAL_EPISODE_MAX_STEPS=500
VIDEO_OUTPUT="true"
VIDEO_IMAGE_SIZE=512
N_ODE_STEPS=10
EXECUTE_HORIZON=8
# -------------------------------------------------------------------------------------------------

TASK_NAME="${TASK_NAME:-$ENV_NAME}"
EVAL_DETERMINISTIC="false"
VIDEO_OUTPUT="${VIDEO_OUTPUT:-false}"

EXTRA_ARGS=("$@")

export ROOT_DIR
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

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

CHECKPOINT_DIR="$(dirname "${CHECKPOINT}")"
if [[ "$(basename "${CHECKPOINT_DIR}")" == "checkpoints" ]]; then
  RUN_DIR="$(dirname "${CHECKPOINT_DIR}")"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_DIR}/eval}"
else
  echo "[ERROR] CHECKPOINT must live under a run's checkpoints/ directory: ${CHECKPOINT}" >&2
  exit 1
fi

echo "[eval] checkpoint=${CHECKPOINT}"
echo "[eval] env=${ENV_NAME} task=${TASK_NAME} episodes=${EPISODES} video_output=${VIDEO_OUTPUT}"
echo "[eval] results_dir=${ROOT_DIR}/${OUTPUT_ROOT}"

PY_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --env-name "${ENV_NAME}"
  --task-name "${TASK_NAME}"
  --episodes "${EPISODES}"
  --episode-max-steps "${EVAL_EPISODE_MAX_STEPS}"
  --output-root "${OUTPUT_ROOT}"
  --video-output "${VIDEO_OUTPUT}"
  --video-height "${VIDEO_IMAGE_SIZE}"
  --video-width "${VIDEO_IMAGE_SIZE}"
  --execute-horizon "${EXECUTE_HORIZON}"
  --n-ode-steps "${N_ODE_STEPS}"
)

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
