#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dodo/Documents/DAggar/robosuite"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
EVAL_MODULE="robosuite.policy.flow_multi_update.eval_flow"

# Checkpoint root for flow-10 training runs.
CKPT_DIR="${ROOT}/checkpoints/multitask_6/policy/flow-update"
RESULTS_DIR="${CKPT_DIR}/results"
LOG_DIR="${RESULTS_DIR}/logs"

# Cartesian product: TRAIN_EPOCHS x TASKS_TO_EVAL.
TASKS_TO_EVAL=(
  "Stack"
  "Lift"
  "NutAssemblyRound"
  "NutAssemblySquare"
  "PickPlaceCereal"
  "PickPlaceMilk"
  "PickPlaceCan"
  "PickPlaceBread"
)
TRAIN_EPOCHS=(100 150 200 300 400)

# GPUs and per-GPU concurrency limit.
GPU_IDS=(0 1)
MAX_PROC_PER_GPU=4
MAX_PARALLEL_JOBS=$(( ${#GPU_IDS[@]} * MAX_PROC_PER_GPU ))

# Eval hyperparameters.
MAX_STEPS="${MAX_STEPS:-500}"
N_ODE_STEPS="${N_ODE_STEPS:-10}"
SUCC_RATE_EPISODES="${SUCC_RATE_EPISODES:-50}"
EPISODES="${EPISODES:-5}"          # 0 = skip video rollouts, only succ_rate
SKIP_EXISTING="${SKIP_EXISTING:-true}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

mkdir -p "${LOG_DIR}"
cd "${ROOT}"

launch_index=0
running_jobs=0
launched_jobs=0
skipped_jobs=0

launch_eval_job() {
  local epoch="$1"
  local task_name="$2"
  local ckpt_path="${CKPT_DIR}/flow_multi_ep$(printf '%04d' "${epoch}").pt"
  local output_dir="${RESULTS_DIR}/ep${epoch}-${task_name}"
  local log_path="${LOG_DIR}/ep${epoch}-${task_name}.log"
  local gpu="${GPU_IDS[$((launch_index % ${#GPU_IDS[@]}))]}"

  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[skip] missing checkpoint: ${ckpt_path}"
    skipped_jobs=$((skipped_jobs + 1))
    return 0
  fi

  if [[ "${SKIP_EXISTING}" == "true" && -f "${output_dir}/info.json" ]]; then
    echo "[skip] already evaluated: ${output_dir}/info.json"
    skipped_jobs=$((skipped_jobs + 1))
    return 0
  fi

  echo "[launch] epoch=${epoch} task=${task_name} gpu=${gpu} ckpt=${ckpt_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    "${PYTHON_BIN}" -m "${EVAL_MODULE}" \
    eval.ckpt="${ckpt_path}" \
    eval.task_name="${task_name}" \
    eval.output_dir="${output_dir}" \
    eval.device="cuda" \
    eval.episodes="${EPISODES}" \
    eval.max_steps="${MAX_STEPS}" \
    eval.succ_rate=true \
    eval.succ_rate_episodes="${SUCC_RATE_EPISODES}" \
    eval.n_ode_steps="${N_ODE_STEPS}" \
    >"${log_path}" 2>&1 &

  launch_index=$((launch_index + 1))
  launched_jobs=$((launched_jobs + 1))
  running_jobs=$((running_jobs + 1))

  if [[ "${running_jobs}" -ge "${MAX_PARALLEL_JOBS}" ]]; then
    wait -n || true
    running_jobs=$((running_jobs - 1))
  fi
}

echo "[config] ckpt_dir=${CKPT_DIR}"
echo "[config] results_dir=${RESULTS_DIR}"
echo "[config] tasks=${TASKS_TO_EVAL[*]}"
echo "[config] epochs=${TRAIN_EPOCHS[*]}"
echo "[config] gpus=${GPU_IDS[*]} max_proc_per_gpu=${MAX_PROC_PER_GPU} max_parallel=${MAX_PARALLEL_JOBS}"

for epoch in "${TRAIN_EPOCHS[@]}"; do
  for task_name in "${TASKS_TO_EVAL[@]}"; do
    launch_eval_job "${epoch}" "${task_name}"
  done
done

wait || true

echo "[done] launched=${launched_jobs} skipped=${skipped_jobs}"
echo "[done] logs: ${LOG_DIR}"
