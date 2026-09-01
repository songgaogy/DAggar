#!/usr/bin/env bash
# Evaluate a published policy checkpoint resolved from <run_root>/<round>.
set -euo pipefail


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
RUN_ROOT="${RUN_ROOT:-outputs/dipole/PickPlaceCereal/w0_explore_20260804_233553}"
ROUND="${ROUND:-001}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/rounds/${ROUND}/policy/checkpoints/latest.pt}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${RUN_ROOT}/inputs/checkpoints/base_policy.pt}"
TASK="${TASK:-PickPlaceCereal}"
OMEGA_VALUES=(0 0.1 0.2 0.5 1.0 2.0)
EPISODES="${EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-500}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
EVAL_GPU="${EVAL_GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_LOCK_FILE="${EVAL_LOCK_FILE:-/tmp/robosuite_dipole_policy_eval.lock}"

cd "$ROOT_DIR"
export MUJOCO_GL="egl"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU}"

exec 9>"${EVAL_LOCK_FILE}"
flock -x 9
echo "[eval] acquired evaluation lock: ${EVAL_LOCK_FILE}"

for omega in "${OMEGA_VALUES[@]}"; do
    echo ">>> Evaluating policy with omega = $omega"
    eval_args=(
      "$PY" -m robosuite.pipeline.modules.evaluation.policy
      --checkpoint "$CHECKPOINT"
      --init-checkpoint "$INIT_CHECKPOINT"
      --env-name "$TASK"
      --task-name "$TASK"
      --omega "$omega"
      --episodes "$EPISODES"
      --episode-max-steps "$MAX_STEPS"
      --video-output "$SAVE_VIDEO"
      --device "$DEVICE"
    )
    "${eval_args[@]}"
    echo
done
