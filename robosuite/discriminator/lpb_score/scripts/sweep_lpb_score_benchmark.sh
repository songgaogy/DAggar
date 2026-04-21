#!/usr/bin/env bash
set -euo pipefail

# Sweep wrapper around `run_lpb_score_benchmark.sh`.
#
# Usage:
#   bash robosuite/discriminator/lpb_score/scripts/sweep_lpb_score_benchmark.sh
#
# Optional overrides (propagated to the underlying runner):
#   DSM_CKPT=... TASKS="..." DEVICE=cuda FEATURE_BATCH_SIZE=256 FAIL_ROOT=... SUCCESS_ROOT=...
#   EVAL_BASE=...           # where to write eval/<run_name>/benchmark.json
#   SWEEP_TAG=grid_v1       # subfolder-ish tag in RUN_NAME
#   DRY_RUN=1               # print commands only
#   GPU_IDS="0 1"           # space-separated CUDA device ids to use
#   MAX_PARALLEL=2          # max concurrent runs (default: number of GPU_IDS)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_lpb_score_benchmark.sh"

if [[ ! -f "${RUNNER}" ]]; then
  echo "[sweep_lpb_score_benchmark] Missing runner: ${RUNNER}" >&2
  exit 1
fi

SWEEP_TAG="${SWEEP_TAG:-sweep_tail}"
DRY_RUN="${DRY_RUN:-}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

ts="$(date +%Y%m%d_%H%M%S)"

# Default output directory for this sweep (can be overridden via env EVAL_BASE).
EVAL_BASE="${EVAL_BASE:-checkpoints/multitask_6/lpb_dipole-new-v6/eval/sweep}"

_jobs_running() {
  jobs -pr | wc -l | tr -d ' '
}

_wait_for_slot() {
  local max_parallel="$1"
  while [[ "$(_jobs_running)" -ge "${max_parallel}" ]]; do
    sleep 1
  done
}

run_case() {
  local case_id="$1"
  local lambda_mode="$2"
  local window="$3"
  local alpha_state="$4"
  local alpha_dyn="$5"
  local beta_state="$6"
  local beta_dyn="$7"
  local delta="$8"
  local purpose="$9"
  local gpu_id="${10}"

  local run_name
  run_name="$(printf "%s_%s/case%02d__mode-%s__win-%s__as-%s__ad-%s__bs-%s__bd-%s__dlt-%s" \
    "${SWEEP_TAG}" "${ts}" "${case_id}" \
    "${lambda_mode}" "${window}" \
    "${alpha_state}" "${alpha_dyn}" "${beta_state}" "${beta_dyn}" \
    "${delta}")"

  echo ""
  echo "[sweep_lpb_score_benchmark] case=${case_id} gpu=${gpu_id} purpose=${purpose}"
  echo "[sweep_lpb_score_benchmark] RUN_NAME=${run_name}"

  if [[ -n "${DRY_RUN}" ]]; then
    echo "DRY_RUN=1 CUDA_VISIBLE_DEVICES='${gpu_id}' RUN_NAME='${run_name}' LAMBDA_MODE='${lambda_mode}' LAMBDA_WINDOW_SIZE='${window}' \\"
    echo "  ALPHA_STATE='${alpha_state}' ALPHA_DYNAMICS='${alpha_dyn}' BETA_STATE='${beta_state}' BETA_DYNAMICS='${beta_dyn}' DELTA='${delta}' \\"
    echo "  bash '${RUNNER}'"
    return 0
  fi

  # Isolate each run to a single GPU.
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  RUN_NAME="${run_name}" \
  LAMBDA_MODE="${lambda_mode}" \
  LAMBDA_WINDOW_SIZE="${window}" \
  ALPHA_STATE="${alpha_state}" \
  ALPHA_DYNAMICS="${alpha_dyn}" \
  BETA_STATE="${beta_state}" \
  BETA_DYNAMICS="${beta_dyn}" \
  DELTA="${delta}" \
  bash "${RUNNER}"
}

IFS=' ' read -r -a GPU_ID_ARR <<< "${GPU_IDS}"
if [[ "${#GPU_ID_ARR[@]}" -le 0 ]]; then
  echo "[sweep_lpb_score_benchmark] GPU_IDS is empty" >&2
  exit 1
fi
if [[ -z "${MAX_PARALLEL}" ]]; then
  MAX_PARALLEL="${#GPU_ID_ARR[@]}"
fi
if [[ "${MAX_PARALLEL}" -le 0 ]]; then
  echo "[sweep_lpb_score_benchmark] MAX_PARALLEL must be >= 1" >&2
  exit 1
fi

echo "[sweep_lpb_score_benchmark] GPU_IDS=${GPU_IDS}"
echo "[sweep_lpb_score_benchmark] MAX_PARALLEL=${MAX_PARALLEL}"

declare -a PIDS=()
declare -a PID_LABELS=()
_launch() {
  local case_id="$1"; shift
  local gpu_id="${GPU_ID_ARR[$(( (case_id - 1) % ${#GPU_ID_ARR[@]} ))]}"
  _wait_for_slot "${MAX_PARALLEL}"
  ( run_case "${case_id}" "$@" "${gpu_id}" ) &
  local pid="$!"
  PIDS+=("${pid}")
  PID_LABELS+=("case${case_id}@gpu${gpu_id}")
  echo "[sweep_lpb_score_benchmark] launched ${PID_LABELS[-1]} pid=${pid}"
}

# Tail sweep (行动 2). Built on top of sweep_20260421_170146 findings:
#   - alpha=0, beta=1 only (alpha branch has no signal on V6).
#   - best traj: mean, win=40 (0.9692). best frame/event: max, win=-1 (frame 0.923, evP 0.984).
#   - Extend max windows {60, 80, 120}, mean windows {50, 60, 80}, and tighten DELTA on max+full.
# Grid (all reuse the same V6 checkpoint set in the runner by default).
# args: case_id mode window  αs   αd   βs   βd  delta  purpose
_launch  11 "max"  "60"  "0"  "0"  "1"  "1"  "5"  "max mid-long window"
_launch  12 "max"  "80"  "0"  "0"  "1"  "1"  "5"  "max mid-long window"
_launch  13 "max"  "120" "0"  "0"  "1"  "1"  "5"  "max mid-long window"
_launch  14 "mean" "50"  "0"  "0"  "1"  "1"  "5"  "mean mid-long window"
_launch  15 "mean" "60"  "0"  "0"  "1"  "1"  "5"  "mean mid-long window"
_launch  16 "mean" "80"  "0"  "0"  "1"  "1"  "5"  "mean mid-long window"
_launch  17 "max"  "-1"  "0"  "0"  "1"  "1"  "3"  "max+cumax stricter threshold (~97th pct)"
_launch  18 "max"  "-1"  "0"  "0"  "1"  "1"  "1"  "max+cumax strictest threshold (~99th pct)"

fail=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  label="${PID_LABELS[$i]}"
  if wait "${pid}"; then
    echo "[sweep_lpb_score_benchmark] finished ${label}"
  else
    echo "[sweep_lpb_score_benchmark] FAILED ${label}" >&2
    fail=1
  fi
done
if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi

echo ""
echo "[sweep_lpb_score_benchmark] done"
