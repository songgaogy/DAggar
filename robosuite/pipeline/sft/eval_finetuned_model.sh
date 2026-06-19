#!/usr/bin/env bash
set -euo pipefail

cd /home/dodo/Documents/DAggar/robosuite

PYTHON="${PYTHON:-/home/dodo/miniconda3/envs/dagger/bin/python}"
TASK="NutAssemblySquare"
SFT_STEPS=2500    # related to naming
VIDEO_OUTPUT="true"
CKPT="outputs/flow-sft/NutAssemblySquare_30pretrain_add-expert-10/flow_offline_NutAssemblySquare_traj00040_steps00005000_at00002500.pt"

EPISODES="${EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-500}"
DEVICE="${DEVICE:-cuda:0}"
N_ODE_STEPS="${N_ODE_STEPS:-10}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-8}"
SEED="${SEED:-42}"
EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-false}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES=0

PY_ARGS=(
  "${CKPT}"
  --env "${TASK}"
  --episodes "${EPISODES}"
  --max-steps "${MAX_STEPS}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --execute-horizon "${EXECUTE_HORIZON}"
  --n-ode-steps "${N_ODE_STEPS}"
  --postfix "${SFT_STEPS}"
)

if [[ "${EVAL_DETERMINISTIC}" == "true" ]]; then
  PY_ARGS+=(--deterministic)
fi

if [[ "${VIDEO_OUTPUT}" == "true" ]]; then
  PY_ARGS+=(--save-video)
fi

"${PYTHON}" -m robosuite.pipeline.sft.eval_flow_offline "${PY_ARGS[@]}" "$@"
