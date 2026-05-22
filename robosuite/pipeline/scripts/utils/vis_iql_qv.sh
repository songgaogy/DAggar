#!/usr/bin/env bash
# Visualize IQL Q/V on HDF5 rollout windows (see algorithms/q_learning/utils/vis_qv.py).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"

IQL_CKPT="${IQL_CKPT:-outputs/DIPOLE_rl/iql_qv_cache/PickPlaceCereal/iql_state.pt}"
TASK_DATA_NAME="${TASK_DATA_NAME:-PickPlaceCereal}"
SPLIT="${SPLIT:-fail_rollout}"
DISC_CKPT="${DISC_CKPT:-checkpoints/lpb_v2/bce_viz_robosuite/viz_bce_PickPlaceMilk-20260518_014308/checkpoints/bce_head.pth}"

exec "$PY" -m robosuite.pipeline.algorithms.q_learning.utils.vis_qv \
    --iql-ckpt "${IQL_CKPT}" \
    --task-data-name "${TASK_DATA_NAME}" \
    --disc-ckpt "${DISC_CKPT}" \
    --split "${SPLIT}" \
    "$@"
