#!/usr/bin/env bash
# Evaluate a published policy checkpoint resolved from <run_root>/<round>.
set -euo pipefail


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
RUN_ROOT="outputs/dipole/PickPlaceCereal/w0_explore_20260804_233553"
ROUND="000"
CHECKPOINT="${RUN_ROOT}/rounds/${ROUND}/policy/checkpoints/latest.pt"
INIT_CHECKPOINT="${RUN_ROOT}/inputs/checkpoints/base_policy.pt"
TASK="PickPlaceCereal"
OMEGA=(0 0.1 0.2 0.5 1.0 2.0)
EPISODES=50
MAX_STEPS=500
SAVE_VIDEO=true
DEVICE="cuda:0"

cd "$ROOT_DIR"
export MUJOCO_GL="egl"
export CUDA_VISIBLE_DEVICES=0

for OMEGA in "${OMEGA[@]}"; do
    echo ">>> Evaluating policy with omega = $OMEGA"
    "$PY" -m robosuite.pipeline.modules.evaluation.policy \
      --checkpoint "$CHECKPOINT" \
      --init-checkpoint "$INIT_CHECKPOINT" \
      --env-name "$TASK" \
      --task-name "$TASK" \
      --omega "$OMEGA" \
      --episodes "$EPISODES" \
      --episode-max-steps "$MAX_STEPS" \
      --video-output "$SAVE_VIDEO" \
      --device "$DEVICE"
    echo
done
