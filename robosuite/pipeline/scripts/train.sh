#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
RUN_ROOT="outputs/dipole/PickPlaceCereal/w0_explore_20260804_233553"
STAGE="all"   # "all" or "disc" -> "vast" -> "policy"
ROUND_INDEX=""   # if "", then continue; else, copy and retrain
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES=1
export HYDRA_FULL_ERROR=1
export MUJOCO_GL="egl"

TRAIN_ARGS=("${RUN_ROOT}" "${STAGE}")
if [[ -n "${ROUND_INDEX}" ]]; then
  TRAIN_ARGS+=(--round-index "${ROUND_INDEX}")
fi

"${PY}" -m robosuite.pipeline.train "${TRAIN_ARGS[@]}"
