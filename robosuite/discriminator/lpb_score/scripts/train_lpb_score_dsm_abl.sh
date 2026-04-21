#!/usr/bin/env bash
set -euo pipefail

# Parallel launcher for the Phase-A × Phase-B training ablation defined in
#   robosuite/discriminator/lpb_score/results/ablation_plan_sigma_loss.md
#
# All runs fix EMA=on (decay 0.999) and dropout=0.1; per-trajectory fail
# labels are preserved (no frame-level label denoising).
#
# Usage:
#   bash robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm_abl.sh
#
# Tunables (env):
#   MAX_TASK_PRL   concurrent runs (default 4 = 2 per GPU when GPU_IDS has 2 GPUs)
#   GPU_IDS        space-separated CUDA ids to round-robin (default "0 1")
#   SIGMA_STAR     σ used by Phase B runs (default 0.08; change after Phase A is analyzed)
#   CASES          subset of case ids to run (default "A1 A2 A3 A4 B1 B2 B3")
#   DRY_RUN=1      print commands only, do not launch
#   PYTHON_BIN     python binary (default daggar env)
#   CKPT           flow policy ckpt
#   BASE_SAVE_DIR  parent of per-run save dirs (default checkpoints/multitask_6/lpb_score_abl)
#   SEED, EPOCHS, BATCH_SIZE, LR, POSITIVE_RATIO, NUM_POS, NUM_NEG, HORIZON,
#   IMAGE_SIZE, ENCODER_BATCH_SIZE, STD_CLAMP_MIN, NUM_WORKERS  passthrough
#
# Output layout (per run):
#   ${BASE_SAVE_DIR}/${SWEEP_TAG}_${ts}/${case_id}__sig-<σ>__lw-<state>-<dyn>/
#       lpb_score_dsm_<case>.pt
#       lpb_score_dsm_<case>_epXXXX.pt
#       .hydra/  console.log  etc.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"
TRAIN_ENTRY="${ROOT}/robosuite/discriminator/lpb_score/train.py"

if [[ ! -f "${TRAIN_ENTRY}" ]]; then
  echo "[train_lpb_score_dsm_abl] Missing train entry: ${TRAIN_ENTRY}" >&2
  exit 1
fi

# --- Defaults that mirror train_lpb_score_dsm.sh (which produced V6) --------
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-16}"
NUM_POS="${NUM_POS:-200}"
NUM_NEG="${NUM_NEG:-100}"
POSITIVE_RATIO="${POSITIVE_RATIO:-0.7}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-2e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
HORIZON="${HORIZON:-2}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-128}"
STD_CLAMP_MIN="${STD_CLAMP_MIN:-0.05}"

# --- Ablation knobs ---------------------------------------------------------
MAX_TASK_PRL="${MAX_TASK_PRL:-4}"
GPU_IDS="${GPU_IDS:-0 1}"
SIGMA_STAR="${SIGMA_STAR:-0.08}"
CASES="${CASES:-A1 A2 A3 A4 B1 B2 B3}"
DRY_RUN="${DRY_RUN:-}"
SWEEP_TAG="${SWEEP_TAG:-abl_sig_loss}"

BASE_SAVE_DIR="${BASE_SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_score_new-v6}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[train_lpb_score_dsm_abl] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

IFS=' ' read -r -a GPU_ID_ARR <<< "${GPU_IDS}"
if [[ "${#GPU_ID_ARR[@]}" -le 0 ]]; then
  echo "[train_lpb_score_dsm_abl] GPU_IDS is empty" >&2
  exit 1
fi
if [[ "${MAX_TASK_PRL}" -le 0 ]]; then
  echo "[train_lpb_score_dsm_abl] MAX_TASK_PRL must be >= 1" >&2
  exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="${BASE_SAVE_DIR}/${SWEEP_TAG}_${ts}"
mkdir -p "${SWEEP_DIR}"

echo "[train_lpb_score_dsm_abl] ROOT=${ROOT}"
echo "[train_lpb_score_dsm_abl] sweep_dir=${SWEEP_DIR}"
echo "[train_lpb_score_dsm_abl] gpu_ids=${GPU_IDS}  max_task_prl=${MAX_TASK_PRL}"
echo "[train_lpb_score_dsm_abl] sigma_star=${SIGMA_STAR}  cases=${CASES}"
echo "[train_lpb_score_dsm_abl] ckpt=${CKPT}"

# --- Run table --------------------------------------------------------------
# id | sigma  | state_lw | dyn_lw | purpose
# Phase B rows resolve sigma later via SIGMA_STAR substitution.
declare -A CASE_SIGMA=(
  [A1]="0.04"        [A2]="0.08"        [A3]="0.12"        [A4]="0.20"
  [B1]="${SIGMA_STAR}"  [B2]="${SIGMA_STAR}"  [B3]="${SIGMA_STAR}"
)
declare -A CASE_STATE_LW=(
  [A1]="1.0"  [A2]="1.0"  [A3]="1.0"  [A4]="1.0"
  [B1]="1.0"  [B2]="0.0"  [B3]="1.0"
)
declare -A CASE_DYN_LW=(
  [A1]="1.0"  [A2]="1.0"  [A3]="1.0"  [A4]="1.0"
  [B1]="0.0"  [B2]="1.0"  [B3]="1.0"
)
declare -A CASE_PURPOSE=(
  [A1]="sigma=0.04 both_losses"
  [A2]="sigma=0.08 both_losses (V6 + EMA/dropout baseline)"
  [A3]="sigma=0.12 both_losses"
  [A4]="sigma=0.20 both_losses"
  [B1]="state-only at sigma=${SIGMA_STAR}"
  [B2]="dynamics-only at sigma=${SIGMA_STAR}"
  [B3]="both at sigma=${SIGMA_STAR} (dup of A2 if SIGMA_STAR=0.08)"
)

# --- Parallel dispatch ------------------------------------------------------
_jobs_running() { jobs -pr | wc -l | tr -d ' '; }
_wait_for_slot() {
  while [[ "$(_jobs_running)" -ge "${MAX_TASK_PRL}" ]]; do
    sleep 2
  done
}

run_case() {
  local case_id="$1"
  local gpu_id="$2"
  local sigma="${CASE_SIGMA[${case_id}]}"
  local state_lw="${CASE_STATE_LW[${case_id}]}"
  local dyn_lw="${CASE_DYN_LW[${case_id}]}"
  local purpose="${CASE_PURPOSE[${case_id}]}"

  local run_name
  run_name="$(printf "%s__sig-%s__lw-%s-%s" "${case_id}" "${sigma}" "${state_lw}" "${dyn_lw}")"
  local run_dir="${SWEEP_DIR}/${run_name}"
  local save_name="lpb_score_dsm_${case_id}.pt"

  echo ""
  echo "[train_lpb_score_dsm_abl] case=${case_id} gpu=${gpu_id} purpose=${purpose}"
  echo "[train_lpb_score_dsm_abl] run_dir=${run_dir}"

  local cmd=(
    "${PYTHON_BIN}" "${TRAIN_ENTRY}"
    "seed=${SEED}"
    "hydra.run.dir=${run_dir}"
    "save_name=${save_name}"
    "save_dir=${run_dir}"
    "policy.ckpt=${CKPT}"
    "policy.encoder_batch_size=${ENCODER_BATCH_SIZE}"
    "data.image_size=${IMAGE_SIZE}"
    "data.transition_horizon=${HORIZON}"
    "data.splits.train.num_pos_traj=${NUM_POS}"
    "data.splits.train.num_neg_traj=${NUM_NEG}"
    "model.std_clamp_min=${STD_CLAMP_MIN}"
    "model.noise_scale=${sigma}"
    "model.state_loss_weight=${state_lw}"
    "model.dynamics_loss_weight=${dyn_lw}"
    "model.dropout=0.1"
    "training.batch_size=${BATCH_SIZE}"
    "training.num_workers=${NUM_WORKERS}"
    "training.epochs=${EPOCHS}"
    "training.lr=${LR}"
    "training.positive_ratio=${POSITIVE_RATIO}"
    "training.use_ema=true"
    "training.ema_decay=0.999"
    "training.val_use_ema=true"
    "training.save_ema_in_checkpoint=true"
  )

  if [[ -n "${DRY_RUN}" ]]; then
    echo "DRY_RUN=1 CUDA_VISIBLE_DEVICES='${gpu_id}' \\"
    printf '  %q ' "${cmd[@]}"
    echo ""
    return 0
  fi

  mkdir -p "${run_dir}"
  local log_file="${run_dir}/console.log"
  # Isolate each run to a single GPU; tee stdout/stderr to console.log.
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${cmd[@]}" > "${log_file}" 2>&1
  )
}

IFS=' ' read -r -a CASE_ARR <<< "${CASES}"

declare -a PIDS=()
declare -a PID_LABELS=()

idx=0
for case_id in "${CASE_ARR[@]}"; do
  if [[ -z "${CASE_SIGMA[${case_id}]+set}" ]]; then
    echo "[train_lpb_score_dsm_abl] Unknown case_id=${case_id}; valid: A1 A2 A3 A4 B1 B2 B3" >&2
    exit 1
  fi
  gpu_id="${GPU_ID_ARR[$(( idx % ${#GPU_ID_ARR[@]} ))]}"
  _wait_for_slot
  run_case "${case_id}" "${gpu_id}" &
  pid="$!"
  PIDS+=("${pid}")
  PID_LABELS+=("${case_id}@gpu${gpu_id}")
  echo "[train_lpb_score_dsm_abl] launched ${PID_LABELS[-1]} pid=${pid}"
  idx=$(( idx + 1 ))
done

fail=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  label="${PID_LABELS[$i]}"
  if wait "${pid}"; then
    echo "[train_lpb_score_dsm_abl] finished ${label}"
  else
    echo "[train_lpb_score_dsm_abl] FAILED ${label}" >&2
    fail=1
  fi
done

echo ""
echo "[train_lpb_score_dsm_abl] sweep_dir=${SWEEP_DIR}"
if [[ "${fail}" -ne 0 ]]; then
  echo "[train_lpb_score_dsm_abl] one or more runs FAILED" >&2
  exit 1
fi
echo "[train_lpb_score_dsm_abl] done"
