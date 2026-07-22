#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
TASK="PickPlaceCereal"
RUN="new"
RUN_NAME="run"
cd "${ROOT_DIR}"

export HYDRA_FULL_ERROR=1
export MUJOCO_GL="glfw"

"${PY}" -m robosuite.pipeline.run_online \
  "${RUN}" "${RUN_NAME}" \
  --task "${TASK}"
