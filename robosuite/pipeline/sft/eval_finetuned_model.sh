#!/usr/bin/env bash
set -euo pipefail

cd /home/dodo/Documents/DAggar/robosuite

PYTHON="${PYTHON:-/home/dodo/miniconda3/envs/dagger/bin/python}"
TASK="PickPlaceCereal"
SFT_STEPS=15000    # related to naming
VIDEO_OUTPUT="true"
VIDEO_IMAGE_SIZE=512
CKPT="outputs/flow-sft/PickPlaceCereal_10pretrain_addsr50/flow_offline_PickPlaceCereal_traj00060_steps00015000.pt"

EPISODES="${EPISODES:-50}"
MAX_STEPS="${MAX_STEPS:-500}"
DEVICE="${DEVICE:-cuda:0}"
N_ODE_STEPS="${N_ODE_STEPS:-10}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-8}"
SEED=10086
EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-false}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES=0

echo "[eval] checkpoint=${CKPT}"
echo "[eval] env=${TASK} episodes=${EPISODES} seed=${SEED} deterministic=${EVAL_DETERMINISTIC} video_output=${VIDEO_OUTPUT}"

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
  PY_ARGS+=(--video-size "${VIDEO_IMAGE_SIZE}")
fi

"${PYTHON}" -m robosuite.pipeline.sft.eval_flow_offline "${PY_ARGS[@]}" "$@"
