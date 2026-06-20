#!/usr/bin/env bash
# generate rollout data for flow multi update
# save to data/<task>/success_rollout or data/<task>/fail_rollout
# note: 1. saved data format should be the same as the expert data
#       2. rollout data should add an additional term for each step: is_success (bool, be true if has success before this step)
#       3. success rollout should be also in the same level as fail rollout (MAX_STEPS), but if success, freeze the image and state, action should be set to all zeros
#       4. saved data should have all views

set -euo pipefail

ROOT="${ROOT:-/home/dodo/Documents/DAggar/robosuite}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
GENERATE_MODULE="robosuite.policy.flow_multi_update.generate_rollout_data"

MAX_PROC_GPU="${MAX_PROC_GPU:-4}"
GPU_LIST="${GPU_LIST:-0 1}"
TASK_NAMES_LIST="${TASK_NAMES_LIST:-PickPlaceBread PickPlaceCereal PickPlaceMilk PickPlaceCan Stack NutAssemblySquare NutAssemblyRound}"
SUCCESS_MAX="${SUCCESS_MAX:-100}"
FAIL_MAX="${FAIL_MAX:-150}"
MAX_STEPS="${MAX_STEPS:-400}"   # max steps for each episode; if reaches, stop episode and mark as fail

N_ODE_STEPS="${N_ODE_STEPS:-10}"   # number of ode steps for each action
ACTION_HORIZON="${ACTION_HORIZON:-8}"   # action horizon for each step
RENDER_HEIGHT="${RENDER_HEIGHT:-256}"   # render height
RENDER_WIDTH="${RENDER_WIDTH:-256}"   # render width
IMAGE_SIZE="${IMAGE_SIZE:-128}"   # image size
CKPT="${CKPT:-/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/flow_multi_ep0100.pt}"   # checkpoint
SKIP_EXISTING="${SKIP_EXISTING:-false}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${ROOT}/outputs/flow_multi_rollout/${RUN_ID}/logs}"

read -r -a GPU <<< "${GPU_LIST}"
read -r -a TASK_NAMES <<< "${TASK_NAMES_LIST}"

if [[ "${#GPU[@]}" -eq 0 ]]; then
  echo "[error] GPU_LIST is empty"
  exit 1
fi

if [[ "${#TASK_NAMES[@]}" -eq 0 ]]; then
  echo "[error] TASK_NAMES_LIST is empty"
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[error] missing checkpoint: ${CKPT}"
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

MAX_PARALLEL_JOBS=$(( ${#GPU[@]} * MAX_PROC_GPU ))
launch_index=0
running_jobs=0
launched_jobs=0
skipped_jobs=0
failed_jobs=0

wait_for_one_job() {
  if ! wait -n; then
    failed_jobs=$((failed_jobs + 1))
  fi
  running_jobs=$((running_jobs - 1))
}

launch_rollout_job() {
  local task_name="$1"
  local keep_mode="$2"
  local num_trajs="$3"
  local rollout_dir_name="${keep_mode}_rollout"
  local output_dir="${ROOT}/data/${task_name}/${rollout_dir_name}"
  local log_path="${LOG_DIR}/${task_name}_${keep_mode}.log"
  local gpu="${GPU[$((launch_index % ${#GPU[@]}))]}"

  if [[ "${SKIP_EXISTING}" == "true" ]] && compgen -G "${output_dir}/*.hdf5" >/dev/null; then
    echo "[skip] existing hdf5 found: ${output_dir}"
    skipped_jobs=$((skipped_jobs + 1))
    return 0
  fi

  mkdir -p "${output_dir}"
  echo "[launch] task=${task_name} keep_mode=${keep_mode} num_trajs=${num_trajs} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" -m "${GENERATE_MODULE}" \
    generate.ckpt="${CKPT}" \
    generate.task_name="${task_name}" \
    generate.output_dir="${output_dir}" \
    generate.num_trajs="${num_trajs}" \
    generate.keep_mode="${keep_mode}" \
    generate.max_steps="${MAX_STEPS}" \
    generate.n_ode_steps="${N_ODE_STEPS}" \
    generate.action_horizon="${ACTION_HORIZON}" \
    generate.render_height="${RENDER_HEIGHT}" \
    generate.render_width="${RENDER_WIDTH}" \
    generate.device="cuda" \
    data.image_size="${IMAGE_SIZE}" \
    >"${log_path}" 2>&1 &

  launch_index=$((launch_index + 1))
  launched_jobs=$((launched_jobs + 1))
  running_jobs=$((running_jobs + 1))

  if [[ "${running_jobs}" -ge "${MAX_PARALLEL_JOBS}" ]]; then
    wait_for_one_job
  fi
}

echo "[config] root=${ROOT}"
echo "[config] ckpt=${CKPT}"
echo "[config] tasks=${TASK_NAMES[*]}"
echo "[config] gpus=${GPU[*]} max_proc_gpu=${MAX_PROC_GPU} max_parallel=${MAX_PARALLEL_JOBS}"
echo "[config] success_max=${SUCCESS_MAX} fail_max=${FAIL_MAX} max_steps=${MAX_STEPS}"
echo "[config] n_ode_steps=${N_ODE_STEPS} action_horizon=${ACTION_HORIZON} image_size=${IMAGE_SIZE}"
echo "[config] log_dir=${LOG_DIR}"

for task_name in "${TASK_NAMES[@]}"; do
  launch_rollout_job "${task_name}" "success" "${SUCCESS_MAX}"
  launch_rollout_job "${task_name}" "fail" "${FAIL_MAX}"
done

while [[ "${running_jobs}" -gt 0 ]]; do
  wait_for_one_job
done

echo "[done] launched=${launched_jobs} skipped=${skipped_jobs} failed=${failed_jobs}"
echo "[done] logs: ${LOG_DIR}"

if [[ "${failed_jobs}" -gt 0 ]]; then
  exit 1
fi
