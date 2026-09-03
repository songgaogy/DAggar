#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
# Alternative source: outputs/dipole/PickPlaceCereal/currect_ref_20260901_212515
RUN_ROOT="${RUN_ROOT:-outputs/dipole/NutAssemblySquare/currect_ref_20260902_162750}"
STAGE="all"   # "all" or "disc" -> "vast" -> "policy"
ROUND_INDEX="000"   # if "", then continue; else, roll back in place and retrain
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
export MUJOCO_GL="egl"

TRAIN_ARGS=("${RUN_ROOT}" "${STAGE}")
if [[ -n "${ROUND_INDEX}" ]]; then
  TRAIN_ARGS+=(--round-index "${ROUND_INDEX}")
fi

"${PY}" -m robosuite.pipeline.train "${TRAIN_ARGS[@]}"
