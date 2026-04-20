#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# TODO: update DSM_CKPT after the SσDC refactor retrain; the old uni-dsm three-head checkpoint
# (checkpoints/multitask_6/lpb_dipole-new/lpb_dipole_dsm_20260412_002031) is no longer loadable
# because the action head has been dropped from the architecture.
DSM_CKPT="checkpoints/multitask_6/lpb_dipole-new-v6/lpb_dipole_dsm_20260420_070130/lpb_dipole_dsm_20260420_070130.pt"
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
DELTA="5"
DELTA_STEP="${DELTA_STEP:-0.5}"
LAMBDA_MODE="${LAMBDA_MODE:-mean}"               # mean | max
LAMBDA_WINDOW_SIZE="${LAMBDA_WINDOW_SIZE:--1}"   # -1 = full-prefix aggregation

# SσDC per-branch weights. Defaults replicate the R5 recipe (β-only, state + dynamics).
ALPHA_STATE="${ALPHA_STATE:-0.0}"
ALPHA_DYNAMICS="${ALPHA_DYNAMICS:-0.0}"
BETA_STATE="${BETA_STATE:-1.0}"
BETA_DYNAMICS="${BETA_DYNAMICS:-1.0}"

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
  echo "[run_lpb_score_benchmark] Unsupported LAMBDA_MODE=${LAMBDA_MODE}. Only mean|max are supported." >&2
  exit 1
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
echo "[run_lpb_score_benchmark] alpha_state=${ALPHA_STATE} alpha_dynamics=${ALPHA_DYNAMICS}"
echo "[run_lpb_score_benchmark] beta_state=${BETA_STATE} beta_dynamics=${BETA_DYNAMICS}"
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
  --alpha-state "${ALPHA_STATE}"
  --alpha-dynamics "${ALPHA_DYNAMICS}"
  --beta-state "${BETA_STATE}"
  --beta-dynamics "${BETA_DYNAMICS}"
)

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
