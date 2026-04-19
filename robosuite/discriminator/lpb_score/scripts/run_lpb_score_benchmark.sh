#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# Required: unified DSM checkpoint to benchmark.
DSM_CKPT="checkpoints/multitask_6/lpb_dipole-new/lpb_dipole_dsm_20260412_002031/lpb_dipole_dsm_20260412_002031.pt"
# Optional task filter, space-separated. Example: "PickPlaceCan PickPlaceBread".
TASKS="PickPlaceCan PickPlaceBread PickPlaceCereal PickPlaceMilk"

# Default output layout mirrors visualize: <parent-of-run>/eval/<run_name>/benchmark.json
EVAL_BASE="${EVAL_BASE:-}"
RUN_NAME="${RUN_NAME:-}"
OUT_JSON="${OUT_JSON:-}"

# Benchmark dataset roots.
FAIL_ROOT="${FAIL_ROOT:-${ROOT}/data/utils/fail_rollout}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${ROOT}/data/utils/success_rollout}"

# Runtime and detector settings.
DEVICE="${DEVICE:-cuda}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-256}"
DELTA="${DELTA:-10.0}"
DELTA_STEP="${DELTA_STEP:-0.5}"
LAMBDA_MODE="${LAMBDA_MODE:-mean}"               # current uni-dsm supports: mean | max
LAMBDA_WINDOW_SIZE="${LAMBDA_WINDOW_SIZE:--1}"   # -1 = full-prefix aggregation
SCORE_MODE="t3_weighted_combo"    # t1_positive_energy | t2_negative_margin | t3_weighted_combo

# Shorthand T3 weights. If branch weights below are set, they take precedence.
ALPHA="${ALPHA:-}"
BETA="${BETA:-}"

# Per-branch T3 weights. When unset, the script falls back to the current
# visualize recipe only if neither the shorthand nor the branch overrides are set.
ALPHA_STATE=0
ALPHA_ACTION=0
ALPHA_DYNAMICS=0
BETA_STATE=1
BETA_ACTION=0
BETA_DYNAMICS=1

# Kept for interface parity with uni-dsm-new; ignored on current uni-dsm because ewma is unsupported.
EWMA_ALPHA="${EWMA_ALPHA:-0.9}"

# Optional benchmark dataset caps for quick smoke runs.
MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-}"

if [[ -z "${DSM_CKPT}" ]]; then
  echo "[run_lpb_score_benchmark] Missing DSM_CKPT=/abs/path/to/model.pt" >&2
  exit 1
fi

if [[ ! -f "${DSM_CKPT}" ]]; then
  echo "[run_lpb_score_benchmark] Checkpoint not found: ${DSM_CKPT}" >&2
  exit 1
fi

if [[ "${LAMBDA_MODE}" != "mean" && "${LAMBDA_MODE}" != "max" ]]; then
  echo "[run_lpb_score_benchmark] Unsupported LAMBDA_MODE=${LAMBDA_MODE}. Current uni-dsm only supports mean|max." >&2
  exit 1
fi

if [[ -z "${ALPHA}" && -z "${ALPHA_STATE}" && -z "${ALPHA_ACTION}" && -z "${ALPHA_DYNAMICS}" ]]; then
  ALPHA_STATE="1.0"
  ALPHA_ACTION="0.0"
  ALPHA_DYNAMICS="1.0"
fi
if [[ -z "${BETA}" && -z "${BETA_STATE}" && -z "${BETA_ACTION}" && -z "${BETA_DYNAMICS}" ]]; then
  BETA_STATE="1.0"
  BETA_ACTION="0.0"
  BETA_DYNAMICS="1.0"
fi

DSM_ABS="$(cd "$(dirname "${DSM_CKPT}")" && pwd)/$(basename "${DSM_CKPT}")"
if [[ -z "${EVAL_BASE}" ]]; then
  EVAL_BASE="$(dirname "$(dirname "${DSM_ABS}")")/eval"
fi
if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "${OUT_JSON}" ]]; then
  OUT_JSON="${EVAL_BASE}/${RUN_NAME}/benchmark.json"
fi
mkdir -p "$(dirname "${OUT_JSON}")"

echo "[run_lpb_score_benchmark] dsm_ckpt=${DSM_ABS}"
echo "[run_lpb_score_benchmark] tasks=${TASKS}"
echo "[run_lpb_score_benchmark] score_mode=${SCORE_MODE}"
echo "[run_lpb_score_benchmark] lambda_mode=${LAMBDA_MODE}"
echo "[run_lpb_score_benchmark] out_json=${OUT_JSON}"

CMD=(
  "${PYTHON_BIN}" -m data.utils.benchmark.examples.run_lpb_score
  --dsm-ckpt "${DSM_ABS}"
  --fail-root "${FAIL_ROOT}"
  --success-root "${SUCCESS_ROOT}"
  --save-json "${OUT_JSON}"
  --device "${DEVICE}"
  --feature-batch-size "${FEATURE_BATCH_SIZE}"
  --delta "${DELTA}"
  --delta-step "${DELTA_STEP}"
  --lambda-mode "${LAMBDA_MODE}"
  --lambda-window-size "${LAMBDA_WINDOW_SIZE}"
  --score-mode "${SCORE_MODE}"
  --ewma-alpha "${EWMA_ALPHA}"
)

if [[ -n "${ALPHA}" ]]; then
  CMD+=(--alpha "${ALPHA}")
fi
if [[ -n "${BETA}" ]]; then
  CMD+=(--beta "${BETA}")
fi
if [[ -n "${ALPHA_STATE}" ]]; then
  CMD+=(--alpha-state "${ALPHA_STATE}")
fi
if [[ -n "${ALPHA_ACTION}" ]]; then
  CMD+=(--alpha-action "${ALPHA_ACTION}")
fi
if [[ -n "${ALPHA_DYNAMICS}" ]]; then
  CMD+=(--alpha-dynamics "${ALPHA_DYNAMICS}")
fi
if [[ -n "${BETA_STATE}" ]]; then
  CMD+=(--beta-state "${BETA_STATE}")
fi
if [[ -n "${BETA_ACTION}" ]]; then
  CMD+=(--beta-action "${BETA_ACTION}")
fi
if [[ -n "${BETA_DYNAMICS}" ]]; then
  CMD+=(--beta-dynamics "${BETA_DYNAMICS}")
fi
if [[ -n "${TASKS}" ]]; then
  # shellcheck disable=SC2206
  TASK_ARRAY=(${TASKS})
  CMD+=(--tasks "${TASK_ARRAY[@]}")
fi
if [[ -n "${MAX_FAIL_PER_TASK}" ]]; then
  CMD+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_SUCCESS_PER_TASK}" ]]; then
  CMD+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi

"${CMD[@]}"
