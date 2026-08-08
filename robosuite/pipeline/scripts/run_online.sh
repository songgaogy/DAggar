#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
TASK="PickPlaceCereal"
RUN="outputs/dipole/PickPlaceCereal/base_20260804_232020"  # "new" or existing run root
RUN_NAME=""  # only used when RUN is "new"; timestamp is appended automatically
cd "${ROOT_DIR}"

export HYDRA_FULL_ERROR=1
export MUJOCO_GL="glfw"
export PYTHONUNBUFFERED=1

if [[ "${RUN}" == "new" ]]; then
  LOG_DIR="outputs/dipole/${TASK}"
  LOG_FILE="${LOG_DIR}/${RUN_NAME:-run}_online.log"
  mkdir -p "${LOG_DIR}"
  COMMAND=("${PY}" -m robosuite.pipeline.run_online \
    new "${RUN_NAME}" \
    --task "${TASK}")
else
  LOG_FILE="${RUN}/run_online.log"
  COMMAND=("${PY}" -m robosuite.pipeline.run_online \
    "${RUN}" \
    --task "${TASK}")
fi

"${COMMAND[@]}" 2>&1 \
  | tee -a "${LOG_FILE}" \
  | sed -u '/\[collect\]\[policy\]\[candidates\]/d'
