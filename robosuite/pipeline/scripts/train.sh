#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
RUN_ROOT=""
STAGE="all"
cd "${ROOT_DIR}"

export HYDRA_FULL_ERROR=1
export MUJOCO_GL="egl"

"${PY}" -m robosuite.pipeline.train \
  "${RUN_ROOT}" "${STAGE}"
