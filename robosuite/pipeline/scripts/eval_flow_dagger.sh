#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
export CUDA_VISIBLE_DEVICES=1

# -------------------------------------------------------------------------------------------------
ENV_NAME="NutAssemblySquare"
CHECKPOINT="outputs/flow-DAgger_deterministic_pu-disc/flow_dagger_NutAssemblySquare_2026-06-22_02-08-20_30pretrain_seed42/checkpoints/step_00015000_updates_00012292_ep_00055.pt"
EPISODES=50
EVAL_EPISODE_MAX_STEPS=500
VIDEO_OUTPUT="true"
VIDEO_IMAGE_SIZE=512
N_ODE_STEPS=10
EXECUTE_HORIZON=8
SEED=10086
# -------------------------------------------------------------------------------------------------

TASK_NAME="${TASK_NAME:-$ENV_NAME}"
EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-false}"

EXTRA_ARGS=("$@")

export ROOT_DIR
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

cd "${ROOT_DIR}"

CHECKPOINT_DIR="$(dirname "${CHECKPOINT}")"
RUN_DIR="$(dirname "${CHECKPOINT_DIR}")"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_DIR}/eval_seed${SEED}}"

echo "[eval] checkpoint=${CHECKPOINT}"
echo "[eval] env=${ENV_NAME} task=${TASK_NAME} episodes=${EPISODES} seed=${SEED} deterministic=${EVAL_DETERMINISTIC} video_output=${VIDEO_OUTPUT}"
echo "[eval] results_dir=${ROOT_DIR}/${OUTPUT_ROOT}"

PY_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --env-name "${ENV_NAME}"
  --task-name "${TASK_NAME}"
  --episodes "${EPISODES}"
  --episode-max-steps "${EVAL_EPISODE_MAX_STEPS}"
  --seed "${SEED}"
  --output-root "${OUTPUT_ROOT}"
  --video-output "${VIDEO_OUTPUT}"
  --video-height "${VIDEO_IMAGE_SIZE}"
  --video-width "${VIDEO_IMAGE_SIZE}"
  --execute-horizon "${EXECUTE_HORIZON}"
  --n-ode-steps "${N_ODE_STEPS}"
)

if [[ "${EVAL_DETERMINISTIC}" == "true" ]]; then
  PY_ARGS+=(--deterministic)
fi

python -m robosuite.pipeline.eval_flow_dagger \
  "${PY_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
