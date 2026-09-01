#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
RUN_ROOT="${RUN_ROOT:-outputs/dipole/PickPlaceCereal/multi-explore_20260804_232020}"
STAGE="${STAGE:-policy}"   # "all" or "disc" -> "vast" -> "policy"
ROUND_INDEX="${ROUND_INDEX:-}"   # if "", then continue; else, copy and retrain
TRAIN_GPU="${TRAIN_GPU:-1}"
BRANCH_BETA="${BRANCH_BETA:-2}"
BRANCH_K="${BRANCH_K:-0}"
BRANCH_ETA="${BRANCH_ETA:-0}"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL="egl"

TRAIN_ARGS=("${RUN_ROOT}" "${STAGE}")
if [[ -n "${ROUND_INDEX}" ]]; then
  TRAIN_ARGS+=(--round-index "${ROUND_INDEX}")
fi
if [[ "${STAGE}" == "policy" ]]; then
  TRAIN_ARGS+=(
    --branch-beta "${BRANCH_BETA}"
    --branch-k "${BRANCH_K}"
    --branch-eta "${BRANCH_ETA}"
  )
fi

"${PY}" -m robosuite.pipeline.train "${TRAIN_ARGS[@]}"
