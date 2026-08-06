#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/dodo/Documents/DAggar/robosuite}"
PYTHON="${PYTHON:-/home/dodo/miniconda3/envs/dagger/bin/python}"
TASK="${TASK:-PickPlaceCereal}"
CHECKPOINT="outputs/baseline/awr/PickPlaceCereal/awr_2026-08-06_22-49-05/checkpoints/episode_00000060.pt"

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] Set CHECKPOINT to an AWR checkpoint or run directory." >&2
  exit 1
fi

cd "${ROOT_DIR}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

"${PYTHON}" -m robosuite.pipeline.eval \
  "task=${TASK}" \
  "evaluation.checkpoint=${CHECKPOINT}" \
  "$@"
