#!/usr/bin/env bash
set -euo pipefail

# Default to headless EGL rendering while still allowing explicit overrides.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

# Usage:
#   bash robosuite/pipeline/test/test_cfg_init_policy.sh
#   INIT_CHECKPOINT=/path/to/ckpt.pt ENV_NAME=PickPlaceBread TASK_NAME=PickPlaceBread \
#     EPISODES=20 DEVICE=cuda:0 bash robosuite/pipeline/test/test_cfg_init_policy.sh
#   VIDEO_OUTPUT=true MAX_VIDEOS=2 bash robosuite/pipeline/test/test_cfg_init_policy.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Prefer the project conda python; fallback to current python.
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

# ------------------------------------------------------------------------
# Required checkpoint: flow/flow-dagger init checkpoint used by DIPOLE cfg init policy.
INIT_CHECKPOINT="${INIT_CHECKPOINT:-checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"

# Common defaults (can be overridden by env vars).
ENV_NAME="${ENV_NAME:-PickPlaceCereal}"
TASK_NAME="${TASK_NAME:-PickPlaceCereal}"
EPISODES="${EPISODES:-20}"
EPISODE_MAX_STEPS="${EPISODE_MAX_STEPS:-400}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/DIPOLE/test_cfg_init_policy}"
# ------------------------------------------------------------------------


if [[ -z "${INIT_CHECKPOINT}" ]]; then
  echo "[ERROR] INIT_CHECKPOINT is required."
  echo "Example:"
  echo "  INIT_CHECKPOINT=/abs/path/to/init.pt bash robosuite/pipeline/test/test_cfg_init_policy.sh"
  exit 1
fi

VIDEO_OUTPUT="${VIDEO_OUTPUT:-true}"
VIDEO_CAMERA="${VIDEO_CAMERA:-agentview}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-512}"
VIDEO_WIDTH="${VIDEO_WIDTH:-512}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
RUN_TAG="${RUN_TAG:-cfg_init_eval}"
DETERMINISTIC="${DETERMINISTIC:-false}"

EXTRA_ARGS=()
if [[ "${DETERMINISTIC}" == "true" ]]; then
  EXTRA_ARGS+=("--deterministic")
fi

echo "[INFO] Running cfg-init policy evaluation (pos-only, omega=0)."
echo "[INFO] init_checkpoint=${INIT_CHECKPOINT}"
echo "[INFO] env_name=${ENV_NAME} task_name=${TASK_NAME} episodes=${EPISODES} device=${DEVICE}"
echo "[INFO] headless MUJOCO_GL=${MUJOCO_GL} video_output=${VIDEO_OUTPUT} video_camera=${VIDEO_CAMERA}"

exec "${PYTHON_BIN}" -m robosuite.pipeline.test.test_cfg_init_policy \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --env-name "${ENV_NAME}" \
  --task-name "${TASK_NAME}" \
  --episodes "${EPISODES}" \
  --episode-max-steps "${EPISODE_MAX_STEPS}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --output-root "${OUTPUT_ROOT}" \
  --video-output "${VIDEO_OUTPUT}" \
  --video-camera "${VIDEO_CAMERA}" \
  --video-fps "${VIDEO_FPS}" \
  --video-height "${VIDEO_HEIGHT}" \
  --video-width "${VIDEO_WIDTH}" \
  --max-videos "${MAX_VIDEOS}" \
  --run-tag "${RUN_TAG}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
