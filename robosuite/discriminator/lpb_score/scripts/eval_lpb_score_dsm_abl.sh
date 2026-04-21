#!/usr/bin/env bash
set -euo pipefail

# Eval all ablation checkpoints produced by train_lpb_score_dsm_abl.sh.
# For each of the 7 cases (A1..A4, B1..B3) we run benchmark.json under
#   * cfgB    — Config B (max, win=-1, β-only): default frame/event-optimal
#   * cfgA    — Config A (mean, win=50, β-only): traj-AUROC-optimal
#   * (A-rows only) cfgB_alpha — Config B with α=1 as a diagnostic for
#     whether a larger σ revived the α OOD branch.
# Phase-B runs scale β per-branch to match the trained head.
#
# Usage:
#   SWEEP_DIR=checkpoints/multitask_6/lpb_dipole-new-v6/abl_sig_loss_20260421_203614 \
#     bash robosuite/discriminator/lpb_score/scripts/eval_lpb_score_dsm_abl.sh
#
# Tunables (env):
#   SWEEP_DIR           (required) path to the ablation sweep dir
#   CASES               subset (default "A1 A2 A3 A4 B1 B2 B3")
#   EVAL_TAG            subfolder name (default eval_<ts>)
#   MAX_TASK_PRL        concurrent evals (default 4 = 2 per GPU when GPU_IDS="0 1")
#   GPU_IDS             space-separated CUDA ids (default "0 1")
#   DEVICE              cuda | cpu (default cuda)
#   FEATURE_BATCH_SIZE  (default 256)
#   FAIL_ROOT, SUCCESS_ROOT  benchmark dataset roots (defaults from runner)
#   DRY_RUN=1           print commands only, do not launch
#   INCLUDE_ALPHA_DIAG=0|1  include the α-diagnostic config for A-rows (default 1)
#
# Output layout:
#   ${SWEEP_DIR}/${EVAL_TAG}/<case>__<cfg>/benchmark.json
#   ${SWEEP_DIR}/${EVAL_TAG}/eval_summary.csv  (aggregate table after runs complete)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"
RUNNER="${SCRIPT_DIR}/run_lpb_score_benchmark.sh"
SWEEP_DIR="${SWEEP_DIR:-checkpoints/multitask_6/lpb_dipole-new-v6/abl_sig_loss_20260421_203614}"

if [[ ! -f "${RUNNER}" ]]; then
  echo "[eval_abl] Missing runner: ${RUNNER}" >&2
  exit 1
fi

if [[ -z "${SWEEP_DIR:-}" ]]; then
  echo "[eval_abl] SWEEP_DIR is required (pointed at a train_lpb_score_dsm_abl.sh output dir)" >&2
  exit 1
fi
if [[ ! -d "${SWEEP_DIR}" ]]; then
  echo "[eval_abl] SWEEP_DIR does not exist: ${SWEEP_DIR}" >&2
  exit 1
fi
SWEEP_DIR="$(cd "${SWEEP_DIR}" && pwd)"

MAX_TASK_PRL="${MAX_TASK_PRL:-4}"
GPU_IDS="${GPU_IDS:-0 1}"
CASES="${CASES:-A1 A2 A3 A4 B1 B2 B3}"
DRY_RUN="${DRY_RUN:-}"
INCLUDE_ALPHA_DIAG="${INCLUDE_ALPHA_DIAG:-1}"
DEVICE="${DEVICE:-cuda}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-256}"

ts="$(date +%Y%m%d_%H%M%S)"
EVAL_TAG="${EVAL_TAG:-eval_${ts}}"
EVAL_BASE="${SWEEP_DIR}/${EVAL_TAG}"
mkdir -p "${EVAL_BASE}"

IFS=' ' read -r -a GPU_ID_ARR <<< "${GPU_IDS}"
if [[ "${#GPU_ID_ARR[@]}" -le 0 ]]; then
  echo "[eval_abl] GPU_IDS is empty" >&2
  exit 1
fi
if [[ "${MAX_TASK_PRL}" -le 0 ]]; then
  echo "[eval_abl] MAX_TASK_PRL must be >= 1" >&2
  exit 1
fi

echo "[eval_abl] SWEEP_DIR=${SWEEP_DIR}"
echo "[eval_abl] EVAL_BASE=${EVAL_BASE}"
echo "[eval_abl] cases=${CASES}  gpu_ids=${GPU_IDS}  max_task_prl=${MAX_TASK_PRL}"
echo "[eval_abl] include_alpha_diag=${INCLUDE_ALPHA_DIAG}"

# --- Case → checkpoint discovery -------------------------------------------
# Each case dir is <SWEEP_DIR>/<case>__sig-<σ>__lw-<state>-<dyn>/;
# final ckpt is lpb_score_dsm_<case>.pt inside.
resolve_ckpt() {
  local case_id="$1"
  local match
  match="$(find "${SWEEP_DIR}" -maxdepth 1 -type d -name "${case_id}__*" -print -quit)"
  if [[ -z "${match}" ]]; then
    echo ""
    return 0
  fi
  local ckpt="${match}/lpb_score_dsm_${case_id}.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo ""
    return 0
  fi
  echo "${ckpt}"
}

# --- Per-case eval configs --------------------------------------------------
# Each entry: "<cfg_tag> <LAMBDA_MODE> <LAMBDA_WINDOW_SIZE> <αs> <αd> <βs> <βd>"
declare -a EVAL_CFGS_A=(
  "cfgB        max  -1   0 0 1 1"
  "cfgA        mean 50   0 0 1 1"
)
if [[ "${INCLUDE_ALPHA_DIAG}" == "1" ]]; then
  EVAL_CFGS_A+=("cfgB_alpha  max  -1   1 1 1 1")
fi
declare -a EVAL_CFGS_B3=(
  "cfgB        max  -1   0 0 1 1"
  "cfgA        mean 50   0 0 1 1"
)
declare -a EVAL_CFGS_B1=(
  "cfgB_state  max  -1   0 0 1 0"
  "cfgA_state  mean 50   0 0 1 0"
)
declare -a EVAL_CFGS_B2=(
  "cfgB_dyn    max  -1   0 0 0 1"
  "cfgA_dyn    mean 50   0 0 0 1"
)

cfgs_for_case() {
  local case_id="$1"
  case "${case_id}" in
    A1|A2|A3|A4) printf '%s\n' "${EVAL_CFGS_A[@]}"   ;;
    B1)          printf '%s\n' "${EVAL_CFGS_B1[@]}"  ;;
    B2)          printf '%s\n' "${EVAL_CFGS_B2[@]}"  ;;
    B3)          printf '%s\n' "${EVAL_CFGS_B3[@]}"  ;;
    *) echo "[eval_abl] Unknown case_id=${case_id}" >&2; return 1 ;;
  esac
}

# --- Parallel dispatch ------------------------------------------------------
_jobs_running() { jobs -pr | wc -l | tr -d ' '; }
_wait_for_slot() {
  while [[ "$(_jobs_running)" -ge "${MAX_TASK_PRL}" ]]; do
    sleep 1
  done
}

run_eval() {
  local case_id="$1"
  local ckpt="$2"
  local cfg_tag="$3"
  local lambda_mode="$4"
  local window="$5"
  local alpha_state="$6"
  local alpha_dyn="$7"
  local beta_state="$8"
  local beta_dyn="$9"
  local gpu_id="${10}"

  local run_name="${case_id}__${cfg_tag}"
  local out_json="${EVAL_BASE}/${run_name}/benchmark.json"

  echo ""
  echo "[eval_abl] case=${case_id} cfg=${cfg_tag} gpu=${gpu_id}"
  echo "[eval_abl] ckpt=${ckpt}"
  echo "[eval_abl] out=${out_json}"

  if [[ -n "${DRY_RUN}" ]]; then
    echo "DRY_RUN=1 CUDA_VISIBLE_DEVICES='${gpu_id}' \\"
    echo "  DSM_CKPT='${ckpt}' EVAL_BASE='${EVAL_BASE}' RUN_NAME='${run_name}' OUT_JSON='${out_json}' \\"
    echo "  LAMBDA_MODE='${lambda_mode}' LAMBDA_WINDOW_SIZE='${window}' \\"
    echo "  ALPHA_STATE='${alpha_state}' ALPHA_DYNAMICS='${alpha_dyn}' BETA_STATE='${beta_state}' BETA_DYNAMICS='${beta_dyn}' \\"
    echo "  DEVICE='${DEVICE}' FEATURE_BATCH_SIZE='${FEATURE_BATCH_SIZE}' \\"
    echo "  bash '${RUNNER}'"
    return 0
  fi

  mkdir -p "${EVAL_BASE}/${run_name}"
  local log_file="${EVAL_BASE}/${run_name}/console.log"
  (
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    DSM_CKPT="${ckpt}" \
    EVAL_BASE="${EVAL_BASE}" \
    RUN_NAME="${run_name}" \
    OUT_JSON="${out_json}" \
    LAMBDA_MODE="${lambda_mode}" \
    LAMBDA_WINDOW_SIZE="${window}" \
    ALPHA_STATE="${alpha_state}" \
    ALPHA_DYNAMICS="${alpha_dyn}" \
    BETA_STATE="${beta_state}" \
    BETA_DYNAMICS="${beta_dyn}" \
    DEVICE="${DEVICE}" \
    FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE}" \
    ${FAIL_ROOT:+FAIL_ROOT="${FAIL_ROOT}"} \
    ${SUCCESS_ROOT:+SUCCESS_ROOT="${SUCCESS_ROOT}"} \
    bash "${RUNNER}" > "${log_file}" 2>&1
  )
}

# --- Plan and launch --------------------------------------------------------
IFS=' ' read -r -a CASE_ARR <<< "${CASES}"

declare -a PIDS=()
declare -a PID_LABELS=()
declare -a MISSING_CASES=()

idx=0
for case_id in "${CASE_ARR[@]}"; do
  ckpt="$(resolve_ckpt "${case_id}")"
  if [[ -z "${ckpt}" ]]; then
    echo "[eval_abl] WARN: no checkpoint for case=${case_id}; skipping" >&2
    MISSING_CASES+=("${case_id}")
    continue
  fi

  while IFS= read -r cfg_line; do
    [[ -z "${cfg_line}" ]] && continue
    # shellcheck disable=SC2206
    cfg_arr=( ${cfg_line} )
    cfg_tag="${cfg_arr[0]}"
    lmode="${cfg_arr[1]}"
    lwin="${cfg_arr[2]}"
    a_s="${cfg_arr[3]}"
    a_d="${cfg_arr[4]}"
    b_s="${cfg_arr[5]}"
    b_d="${cfg_arr[6]}"

    gpu_id="${GPU_ID_ARR[$(( idx % ${#GPU_ID_ARR[@]} ))]}"
    _wait_for_slot
    run_eval "${case_id}" "${ckpt}" "${cfg_tag}" "${lmode}" "${lwin}" \
             "${a_s}" "${a_d}" "${b_s}" "${b_d}" "${gpu_id}" &
    pid="$!"
    PIDS+=("${pid}")
    PID_LABELS+=("${case_id}:${cfg_tag}@gpu${gpu_id}")
    echo "[eval_abl] launched ${PID_LABELS[-1]} pid=${pid}"
    idx=$(( idx + 1 ))
  done < <(cfgs_for_case "${case_id}")
done

fail=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  label="${PID_LABELS[$i]}"
  if wait "${pid}"; then
    echo "[eval_abl] finished ${label}"
  else
    echo "[eval_abl] FAILED ${label}" >&2
    fail=1
  fi
done

if [[ "${#MISSING_CASES[@]}" -gt 0 ]]; then
  echo "[eval_abl] skipped (no ckpt): ${MISSING_CASES[*]}" >&2
fi

# --- Aggregate to CSV -------------------------------------------------------
summary_csv="${EVAL_BASE}/eval_summary.csv"
if [[ -z "${DRY_RUN}" ]]; then
  echo "[eval_abl] aggregating → ${summary_csv}"
  "${PYTHON_BIN}" - "${EVAL_BASE}" "${summary_csv}" <<'PY'
import csv, json, os, sys
eval_base, out_csv = sys.argv[1], sys.argv[2]
rows = []
for entry in sorted(os.listdir(eval_base)):
    bj = os.path.join(eval_base, entry, "benchmark.json")
    if not os.path.isfile(bj):
        continue
    try:
        with open(bj) as f:
            b = json.load(f)
    except Exception as e:
        print(f"[eval_abl/py] skip {entry}: {e}", file=sys.stderr)
        continue
    tl = b.get("trajectory_level", {})
    sl = b.get("step_level", {})
    argv = b.get("config", {}).get("eval_run", {}).get("argv", [])
    def _pick(flag, default=""):
        for i, a in enumerate(argv):
            if a == flag and i + 1 < len(argv):
                return argv[i + 1]
        return default
    rows.append({
        "run": entry,
        "case": entry.split("__", 1)[0],
        "cfg":  entry.split("__", 1)[1] if "__" in entry else "",
        "lambda_mode":    _pick("--lambda-mode"),
        "lambda_window":  _pick("--lambda-window-size"),
        "alpha_state":    _pick("--alpha-state"),
        "alpha_dyn":      _pick("--alpha-dynamics"),
        "beta_state":     _pick("--beta-state"),
        "beta_dyn":       _pick("--beta-dynamics"),
        "traj_auroc":     f"{tl.get('auroc', float('nan')):.4f}",
        "traj_auprc":     f"{tl.get('auprc', float('nan')):.4f}",
        "best_f1":        f"{tl.get('best_f1', float('nan')):.4f}",
        "frame_auroc":    f"{sl.get('frame_auroc', float('nan')):.4f}",
        "frame_auprc":    f"{sl.get('frame_auprc', float('nan')):.4f}",
        "frame_f1":       f"{sl.get('frame_f1', float('nan')):.4f}",
        "event_precision":f"{sl.get('event_precision', float('nan')):.4f}",
        "event_recall":   f"{sl.get('event_recall', float('nan')):.4f}",
        "num_pred_segs":  sl.get("num_pred_segments", ""),
        "num_gt_segs":    sl.get("num_gt_segments", ""),
        "delay_median":   sl.get("detection_delay_median", ""),
    })
if not rows:
    print("[eval_abl/py] no benchmark.json found under", eval_base, file=sys.stderr)
    sys.exit(0)
cols = list(rows[0].keys())
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)
# Pretty print
widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
fmt = "  ".join(f"{{{c}:<{widths[c]}}}" for c in cols)
print(fmt.format(**{c: c for c in cols}))
print("-" * (sum(widths.values()) + 2 * (len(cols) - 1)))
for r in rows:
    print(fmt.format(**r))
PY
fi

echo ""
echo "[eval_abl] EVAL_BASE=${EVAL_BASE}"
echo "[eval_abl] summary_csv=${summary_csv}"
if [[ "${fail}" -ne 0 ]]; then
  echo "[eval_abl] one or more evals FAILED" >&2
  exit 1
fi
echo "[eval_abl] done"
