#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

TASK_DATA_NAME="${TASK_DATA_NAME:-PickPlaceCereal}"
QV_CACHE="${QV_CACHE:-outputs/awr/qv_cache/${TASK_DATA_NAME}.pt}"
SPLIT="${SPLIT:-success_rollout}"
DEMO_ROOT="${DEMO_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/awr/qv_visualization}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
DEMO_KEY="${DEMO_KEY:-}"
SEED="${SEED:-42}"
MAX_WINDOWS="${MAX_WINDOWS:-}"
DEVICE="${DEVICE:-cuda:0}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda:1}"
VIDEO_FPS="${VIDEO_FPS:-20}"

EXTRA_ARGS=("$@")

export ROOT_DIR
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "${ROOT_DIR}"

if [[ ! -f "${QV_CACHE}" ]]; then
  echo "[ERROR] Q/V cache file does not exist: ${QV_CACHE}" >&2
  exit 1
fi

if [[ ! -d "${DEMO_ROOT}/${TASK_DATA_NAME}/${SPLIT}" ]]; then
  echo "[ERROR] Demo split directory does not exist: ${DEMO_ROOT}/${TASK_DATA_NAME}/${SPLIT}" >&2
  exit 1
fi

PY_ARGS=(
  --qv-cache "${QV_CACHE}"
  --task-data-name "${TASK_DATA_NAME}"
  --split "${SPLIT}"
  --demo-root "${DEMO_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --inference-device "${INFERENCE_DEVICE}"
  --video-fps "${VIDEO_FPS}"
)

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  PY_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

if [[ -n "${DEMO_KEY}" ]]; then
  PY_ARGS+=(--demo-key "${DEMO_KEY}")
fi

if [[ -n "${MAX_WINDOWS}" ]]; then
  PY_ARGS+=(--max-windows "${MAX_WINDOWS}")
fi

echo "[awr_qv] qv_cache=${QV_CACHE}"
echo "[awr_qv] task_data_name=${TASK_DATA_NAME} split=${SPLIT}"
echo "[awr_qv] seed=${SEED} max_windows=${MAX_WINDOWS:-all}"

"${PYTHON_BIN}" -m robosuite.pipeline.test.visualize_awr_init_qv \
  "${PY_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
