#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
TASK="PickPlaceCereal"
RUN="outputs/dipole/PickPlaceCereal/naive_trial_20260802_175238"  # "new" or existing run root
RUN_NAME="naive_trial"  # only used when RUN is "new"; timestamp is appended automatically
cd "${ROOT_DIR}"

export HYDRA_FULL_ERROR=1
export MUJOCO_GL="glfw"

if [[ "${RUN}" == "new" ]]; then
  "${PY}" -m robosuite.pipeline.run_online \
    new "${RUN_NAME}" \
    --task "${TASK}"
else
  "${PY}" -m robosuite.pipeline.run_online \
    "${RUN}" \
    --task "${TASK}"
fi
