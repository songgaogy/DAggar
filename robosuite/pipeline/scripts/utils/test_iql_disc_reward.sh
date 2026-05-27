#!/usr/bin/env bash
# LPB disc reward viz — thin wrapper around test_visualize_lpb_disc_reward.py
#
#   bash robosuite/pipeline/scripts/utils/test_iql_disc_reward.sh
#   ENVIRONMENT=PickPlaceCereal SPLIT=success_rollout DEMO=demo_000012 bash .../test_iql_disc_reward.sh
#
# Env (all optional): ENVIRONMENT, SPLIT, DEMO, LPB_DISC_VIZ_HDF5, DEVICE, OUTPUT_DIR, SKIP_ASSERT

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
cd "$ROOT_DIR"

ENVIRONMENT="PickPlaceMilk"
SPLIT="fail_rollout"     # success_rollout or fail_rollout
DEMO="demo_000001"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/DIPOLE_rl/test}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export LPB_DISC_VIZ_TASK="${ENVIRONMENT}"
export LPB_DISC_VIZ_DEMO="${DEMO}"
export LPB_DISC_VIZ_DEVICE="cuda:0"
export LPB_DISC_VIZ_OUTPUT="${OUTPUT_DIR}/lpb_disc_reward_${ENVIRONMENT}_${SPLIT}_${DEMO}.png"

if [[ -z "${LPB_DISC_VIZ_HDF5:-}" ]]; then
  shopt -s nullglob
  files=( "${ROOT_DIR}/data/${ENVIRONMENT}/${SPLIT}"/*.hdf5 )
  [[ ${#files[@]} -gt 0 ]] || { echo "[ERROR] no HDF5 in data/${ENVIRONMENT}/${SPLIT}" >&2; exit 1; }
  export LPB_DISC_VIZ_HDF5="${files[0]}"
fi

if [[ "${SPLIT}" != "fail_rollout" || "${SKIP_ASSERT:-}" == "1" ]]; then
  export LPB_DISC_VIZ_SKIP_ASSERT=1
fi

"${PY}" -m pytest \
  robosuite/pipeline/algorithms/q_learning/tests/test_visualize_lpb_disc_reward.py -s -v
