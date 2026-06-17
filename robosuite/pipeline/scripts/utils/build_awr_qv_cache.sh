#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
VALUE_WARMUP_STEPS="${VALUE_WARMUP_STEPS:-20000}"
EXPERT_NUM_TRAJ="${EXPERT_NUM_TRAJ:-20}"
SUCCESS_NUM_TRAJ="${SUCCESS_NUM_TRAJ:-180}"
FAIL_NUM_TRAJ="${FAIL_NUM_TRAJ:-180}"
LEARNER_DEVICE="${LEARNER_DEVICE:-cuda:0}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda:1}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
TASK_FILTER="${TASK_FILTER:-all}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
EXTRA_ARGS=("$@")

TASK_SPECS=(
  "Lift:PandaLift"
  "PickPlaceCan:PandaPickPlaceCan"
  "Stack:PandaStack"
  "PickPlaceBread:PickPlaceBread"
  "PickPlaceCereal:PickPlaceCereal"
  "PickPlaceMilk:PickPlaceMilk"
)

cd "${ROOT_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HYDRA_FULL_ERROR=1

# file directory changed; modify before run!
for spec in "${TASK_SPECS[@]}"; do
  IFS=":" read -r ENVIRONMENT DEMO_TASK_NAME <<< "${spec}"
  if [[ "${TASK_FILTER}" != "all" && "${TASK_FILTER}" != "${DEMO_TASK_NAME}" && "${TASK_FILTER}" != "${ENVIRONMENT}" ]]; then
    continue
  fi
  if [[ ! -d "data/${DEMO_TASK_NAME}/fail_rollout" ]] || [[ -z "$(find "data/${DEMO_TASK_NAME}/fail_rollout" -mindepth 1 -maxdepth 1 -type f 2>/dev/null)" ]]; then
    echo "[build_awr_qv_cache][skip] task=${DEMO_TASK_NAME} reason=no_fail_rollout_files"
    continue
  fi

  echo "[build_awr_qv_cache] env=${ENVIRONMENT} task=${DEMO_TASK_NAME}"
  "${PYTHON_BIN}" -m robosuite.pipeline.algorithms.awr.models.build_awr_qv_cache \
    env.environment="${ENVIRONMENT}" \
    data.task_name="${DEMO_TASK_NAME}" \
    data.expert_num_trajectories="${EXPERT_NUM_TRAJ}" \
    data.success_num_trajectories="${SUCCESS_NUM_TRAJ}" \
    data.fail_num_trajectories="${FAIL_NUM_TRAJ}" \
    runtime.init_checkpoint="${INIT_CHECKPOINT}" \
    runtime.qv_cache.force_rebuild="${FORCE_REBUILD}" \
    runtime.discriminator_reward_enabled=false \
    algorithm.trainer.value_warmup_steps="${VALUE_WARMUP_STEPS}" \
    algorithm.awr.device="${LEARNER_DEVICE}" \
    algorithm.awr.inference_device="${INFERENCE_DEVICE}" \
    logging.use_wandb=false \
    intervention.enabled=false \
    discriminator.enabled=false \
    runtime.interactive=false \
    runtime.viewer_enabled=false \
    "${EXTRA_ARGS[@]}"
done
