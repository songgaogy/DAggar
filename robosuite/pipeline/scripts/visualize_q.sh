#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/dodo/Documents/DAggar/robosuite}"
PYTHON="${PYTHON:-/home/dodo/miniconda3/envs/dagger/bin/python}"
CHECKPOINT="${CHECKPOINT:-}"
SUCCESS_ROLLOUT_DIR="${SUCCESS_ROLLOUT_DIR:-data/PickPlaceCereal/success_rollout}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/baseline/hil-serl/PickPlaceCereal/qv_visualization}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-}"
MC_ACTION_SAMPLES="${MC_ACTION_SAMPLES:-32}"
DEVICE="${DEVICE:-cuda:0}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda:0}"
VIDEO_FPS="${VIDEO_FPS:-20}"

cd "${ROOT_DIR}"

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint file does not exist: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${SUCCESS_ROLLOUT_DIR}" ]]; then
  echo "[ERROR] Success rollout directory does not exist: ${SUCCESS_ROLLOUT_DIR}" >&2
  exit 1
fi

export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --success-rollout-dir "${SUCCESS_ROLLOUT_DIR}"
  --output-root "${OUTPUT_ROOT}"
  --seed "${SEED}"
  --mc-action-samples "${MC_ACTION_SAMPLES}"
  --device "${DEVICE}"
  --inference-device "${INFERENCE_DEVICE}"
  --video-fps "${VIDEO_FPS}"
)
if [[ -n "${MAX_STEPS}" ]]; then
  ARGS+=(--max-steps "${MAX_STEPS}")
fi

"${PYTHON}" -m robosuite.pipeline.visualize_q "${ARGS[@]}" "$@"
