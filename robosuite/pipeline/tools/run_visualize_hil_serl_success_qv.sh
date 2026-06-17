#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

CHECKPOINT="outputs/hil_serl/hil_serl_PickPlaceCereal_2026-04-30_14-20-15/checkpoints/step_00030000_updates_00015175_ep_00306.pt"
SUCCESS_ROLLOUT_DIR="${SUCCESS_ROLLOUT_DIR:-data/PickPlaceCereal/success_rollout}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/hil_serl/qv_visualization}"
SEED=0
MAX_STEPS="${MAX_STEPS:-}"
MC_ACTION_SAMPLES="${MC_ACTION_SAMPLES:-32}"
DEVICE="${DEVICE:-cuda:0}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda:0}"
VIDEO_FPS="${VIDEO_FPS:-20}"

EXTRA_ARGS=("$@")

export ROOT_DIR
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "${ROOT_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint file does not exist: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -d "${SUCCESS_ROLLOUT_DIR}" ]]; then
  echo "[ERROR] Success rollout directory does not exist: ${SUCCESS_ROLLOUT_DIR}" >&2
  exit 1
fi

PY_ARGS=(
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
  PY_ARGS+=(--max-steps "${MAX_STEPS}")
fi

echo "[qv] checkpoint=${CHECKPOINT}"
echo "[qv] success_rollout_dir=${SUCCESS_ROLLOUT_DIR}"
echo "[qv] seed=${SEED} max_steps=${MAX_STEPS:-all} mc_action_samples=${MC_ACTION_SAMPLES}"

"${PYTHON_BIN}" -m robosuite.pipeline.test.visualize_hil_serl_success_qv \
  "${PY_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
