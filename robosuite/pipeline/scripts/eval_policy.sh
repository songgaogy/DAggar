#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/dodo/Documents/DAggar/robosuite}"
PYTHON="${PYTHON:-/home/dodo/miniconda3/envs/dagger/bin/python}"
NUM_EPISODES="${NUM_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-500}"

CKPT="outputs/baseline/dsrl/PickPlaceCereal/PickPlaceCereal_2026-09-06_23-10-50/checkpoints/episode_00000200.pt"

cd "${ROOT_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONFAULTHANDLER=1

ARGS=(
  --checkpoint "${CKPT}"
  --num-episodes "${NUM_EPISODES}"
  --max-steps "${MAX_STEPS}"
)
if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "${DEVICE}")
fi

exec "${PYTHON}" -m robosuite.pipeline.eval_policy "${ARGS[@]}" "$@"
