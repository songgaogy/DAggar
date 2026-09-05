#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/dodo/Documents/DAggar/robosuite}"
PYTHON="${PYTHON:-/home/dodo/miniconda3/envs/dagger/bin/python}"
NUM_EPISODES="${NUM_EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-500}"

if [[ -z "${CKPT:-}" ]]; then
  echo "[ERROR] Set CKPT to a DSRL checkpoint file." >&2
  exit 1
fi

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
