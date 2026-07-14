#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

ROOT="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

cd "${ROOT}"

# Single-task eval example. For epoch x task grid eval, use eval_flow_multi_parallel.sh.
"${PYTHON_BIN}" -m robosuite.policy.flow_multi_update.eval_flow \
  eval.ckpt="checkpoints/multitask_6/policy/flow-10/flow_multi_ep0400.pt" \
  eval.task_name="PickPlaceBread" \
  eval.output_dir="${ROOT}/checkpoints/multitask_6/policy/flow-10/results/ep400-PickPlaceBread" \
  eval.max_steps=500 \
  eval.succ_rate=true \
  eval.n_ode_steps=10
