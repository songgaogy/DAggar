#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
VALUE_WARMUP_STEPS="${VALUE_WARMUP_STEPS:-20000}"
EXPERT_NUM_TRAJ="${EXPERT_NUM_TRAJ:-50}"
SUCCESS_NUM_TRAJ="${SUCCESS_NUM_TRAJ:-50}"
FAIL_NUM_TRAJ="${FAIL_NUM_TRAJ:-50}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
TASK_FILTER="${TASK_FILTER:-all}"
MAX_JOBS="${MAX_JOBS:-4}"
LEARNER_DEVICES="${LEARNER_DEVICES:-${LEARNER_DEVICE:-cuda:0}}"
INFERENCE_DEVICES="${INFERENCE_DEVICES:-${INFERENCE_DEVICE:-cuda:0}}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
EXTRA_ARGS=("$@")

TASK_SPECS=(
  "Stack:PandaStack"
  "PickPlaceBread:PickPlaceBread"
  "PickPlaceCereal:PickPlaceCereal"
  "PickPlaceMilk:PickPlaceMilk"
)

IFS="," read -r -a LEARNER_DEVICE_LIST <<< "${LEARNER_DEVICES}"
IFS="," read -r -a INFERENCE_DEVICE_LIST <<< "${INFERENCE_DEVICES}"
IFS="," read -r -a TASK_FILTER_LIST <<< "${TASK_FILTER}"

cd "${ROOT_DIR}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="songgao-personal"
export HYDRA_FULL_ERROR=1

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/outputs/awr/qv_cache_parallel_logs/$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "${LOG_ROOT}"

if (( MAX_JOBS < 1 )); then
  echo "[ERROR] MAX_JOBS must be >= 1, got ${MAX_JOBS}" >&2
  exit 1
fi

if (( ${#LEARNER_DEVICE_LIST[@]} == 0 || ${#INFERENCE_DEVICE_LIST[@]} == 0 )); then
  echo "[ERROR] LEARNER_DEVICES and INFERENCE_DEVICES must not be empty." >&2
  exit 1
fi

task_selected() {
  local environment="$1"
  local demo_task_name="$2"
  local item
  if [[ "${TASK_FILTER}" == "all" ]]; then
    return 0
  fi
  for item in "${TASK_FILTER_LIST[@]}"; do
    item="$(echo "${item}" | xargs)"
    if [[ "${item}" == "${environment}" || "${item}" == "${demo_task_name}" ]]; then
      return 0
    fi
  done
  return 1
}

has_split_files() {
  local split_dir="$1"
  [[ -d "${split_dir}" ]] || return 1
  find "${split_dir}" -mindepth 1 -maxdepth 1 -type f -print -quit | grep -q .
}

run_task() {
  local environment="$1"
  local demo_task_name="$2"
  local learner_device="$3"
  local inference_device="$4"
  local log_file="$5"
  local status_file="$6"

  {
    echo "[qv_cache_parallel][start] env=${environment} task=${demo_task_name} learner=${learner_device} inference=${inference_device}"
    set +e
    "${PYTHON_BIN}" -m robosuite.pipeline.algorithms.awr.models.build_awr_qv_cache \
      env.environment="${environment}" \
      data.task_name="${demo_task_name}" \
      data.expert_num_trajectories="${EXPERT_NUM_TRAJ}" \
      data.success_num_trajectories="${SUCCESS_NUM_TRAJ}" \
      data.fail_num_trajectories="${FAIL_NUM_TRAJ}" \
      runtime.init_checkpoint="${INIT_CHECKPOINT}" \
      runtime.qv_cache.force_rebuild="${FORCE_REBUILD}" \
      runtime.discriminator_reward_enabled=false \
      algorithm.trainer.value_warmup_steps="${VALUE_WARMUP_STEPS}" \
      algorithm.awr.device="${learner_device}" \
      algorithm.awr.inference_device="${inference_device}" \
      logging.use_wandb=false \
      intervention.enabled=false \
      discriminator.enabled=false \
      runtime.interactive=false \
      runtime.viewer_enabled=false \
      "${EXTRA_ARGS[@]}"
    local exit_code=$?
    set -e
    echo "[qv_cache_parallel][done] env=${environment} task=${demo_task_name} exit=${exit_code}"
    echo "${exit_code}" > "${status_file}"
    return "${exit_code}"
  } > "${log_file}" 2>&1
}

declare -a STARTED_TASKS=()
declare -a STARTED_LOGS=()
declare -a STARTED_STATUS_FILES=()
task_index=0

echo "[qv_cache_parallel] log_root=${LOG_ROOT}"
echo "[qv_cache_parallel] max_jobs=${MAX_JOBS} learner_devices=${LEARNER_DEVICES} inference_devices=${INFERENCE_DEVICES}"

for spec in "${TASK_SPECS[@]}"; do
  IFS=":" read -r ENVIRONMENT DEMO_TASK_NAME <<< "${spec}"
  if ! task_selected "${ENVIRONMENT}" "${DEMO_TASK_NAME}"; then
    continue
  fi
  if ! has_split_files "data/${DEMO_TASK_NAME}/success_rollout"; then
    echo "[qv_cache_parallel][skip] task=${DEMO_TASK_NAME} reason=no_success_rollout_files"
    continue
  fi
  if ! has_split_files "data/${DEMO_TASK_NAME}/fail_rollout"; then
    echo "[qv_cache_parallel][skip] task=${DEMO_TASK_NAME} reason=no_fail_rollout_files"
    continue
  fi

  learner_device="${LEARNER_DEVICE_LIST[$((task_index % ${#LEARNER_DEVICE_LIST[@]}))]}"
  inference_device="${INFERENCE_DEVICE_LIST[$((task_index % ${#INFERENCE_DEVICE_LIST[@]}))]}"
  log_file="${LOG_ROOT}/${DEMO_TASK_NAME}.log"
  status_file="${LOG_ROOT}/${DEMO_TASK_NAME}.status"
  rm -f "${status_file}"

  while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
    wait -n || true
  done

  run_task "${ENVIRONMENT}" "${DEMO_TASK_NAME}" "${learner_device}" "${inference_device}" "${log_file}" "${status_file}" &
  STARTED_TASKS+=("${DEMO_TASK_NAME}")
  STARTED_LOGS+=("${log_file}")
  STARTED_STATUS_FILES+=("${status_file}")
  echo "[qv_cache_parallel][launch] task=${DEMO_TASK_NAME} pid=$! log=${log_file}"
  task_index=$((task_index + 1))
done

if (( ${#STARTED_TASKS[@]} == 0 )); then
  echo "[qv_cache_parallel] no tasks launched."
  exit 0
fi

while (( $(jobs -pr | wc -l) > 0 )); do
  wait -n || true
done

failed=0
for idx in "${!STARTED_TASKS[@]}"; do
  task="${STARTED_TASKS[$idx]}"
  log_file="${STARTED_LOGS[$idx]}"
  status_file="${STARTED_STATUS_FILES[$idx]}"
  status="missing"
  if [[ -f "${status_file}" ]]; then
    status="$(cat "${status_file}")"
  fi
  if [[ "${status}" == "0" ]]; then
    echo "[qv_cache_parallel][ok] task=${task} log=${log_file}"
  else
    failed=$((failed + 1))
    echo "[qv_cache_parallel][fail] task=${task} status=${status} log=${log_file}" >&2
  fi
done

if (( failed > 0 )); then
  echo "[qv_cache_parallel] failed_tasks=${failed} log_root=${LOG_ROOT}" >&2
  exit 1
fi

echo "[qv_cache_parallel] all tasks completed successfully. log_root=${LOG_ROOT}"
