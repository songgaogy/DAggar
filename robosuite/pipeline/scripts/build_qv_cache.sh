#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/dodo/Documents/DAggar/robosuite}"
PYTHON="${PYTHON:-/home/dodo/miniconda3/envs/dagger/bin/python}"
TASK="${TASK:-PickPlaceCereal}"

cd "${ROOT_DIR}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

"${PYTHON}" -m robosuite.pipeline.build_qv_cache "task=${TASK}" "$@"
