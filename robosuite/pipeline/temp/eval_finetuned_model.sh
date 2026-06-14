#!/usr/bin/env bash
set -euo pipefail

cd /home/dodo/Documents/DAggar/robosuite

PYTHON="${PYTHON:-/home/dodo/miniconda3/envs/dagger/bin/python}"
TASK="PickPlaceBread"
EPISODES="50"
MAX_STEPS="500"
DEVICE="cuda:1"
N_ODE_STEPS="10"
EXECUTE_HORIZON="8"
CKPT="outputs/flow_dagger_no-hil/PickPlaceBread_expert-pretrain-data/flow_offline_PickPlaceBread_traj00020_steps00010000_at00000000.pt"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

CHECKPOINT_FORMAT="$("${PYTHON}" -c 'import sys, torch; payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False); print("flow_dagger" if "core" in payload else "flow_multi" if ("model" in payload or "ema_model" in payload) else "unknown")' "${CKPT}")"

case "${CHECKPOINT_FORMAT}" in
  flow_multi)
    echo "[eval] checkpoint_format=flow_multi ckpt=${CKPT}"
    "${PYTHON}" robosuite/policy/flow_multi/eval_flow.py \
      eval.ckpt="${CKPT}" \
      eval.task_name="[\"${TASK}\"]" \
      eval.device="${DEVICE}" \
      eval.episodes="${EPISODES}" \
      eval.max_steps="${MAX_STEPS}" \
      eval.succ_rate=true \
      eval.n_ode_steps="${N_ODE_STEPS}" \
      eval.video_dir=""
    ;;
  flow_dagger)
    echo "[eval] checkpoint_format=flow_dagger ckpt=${CKPT}"
    "${PYTHON}" -m robosuite.pipeline.temp.eval_flow_offline \
      "${CKPT}" \
      --env "${TASK}" \
      --episodes "${EPISODES}" \
      --max-steps "${MAX_STEPS}" \
      --device "${DEVICE}" \
      --execute-horizon "${EXECUTE_HORIZON}" \
      --n-ode-steps "${N_ODE_STEPS}"
    ;;
  *)
    echo "Unsupported checkpoint format: ${CKPT}" >&2
    exit 1
    ;;
esac
