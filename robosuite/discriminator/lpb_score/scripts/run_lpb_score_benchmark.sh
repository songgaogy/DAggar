#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

DSM_CKPT="checkpoints/multitask_6/lpb_dipole-new-v5/lpb_dipole_dsm_20260420_011232/lpb_dipole_dsm_20260420_011232_ep0010.pt"
TASKS="PandaPickPlaceCan PickPlaceBread PickPlaceCereal PickPlaceMilk"

# Default layout mirrors visualize: <EVAL_BASE>/run_<timestamp>/benchmark.json
# (visualize uses <save_dir>/run_<timestamp>/...; here EVAL_BASE defaults to ../eval beside the DSM run folder).
EVAL_BASE="${EVAL_BASE:-}"
RUN_NAME="${RUN_NAME:-}"
OUT_JSON="${OUT_JSON:-}"

FAIL_ROOT="${FAIL_ROOT:-${ROOT}/data/utils/fail_rollout}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${ROOT}/data/utils/success_rollout}"
DEVICE="${DEVICE:-cuda}"                        # feature + detector device
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-256}" # batched latent-window scoring size
DELTA="${DELTA:-10.0}"                          # percentile calibration delta for lambda threshold
DELTA_STEP="${DELTA_STEP:-0.5}"                 # adaptive threshold step size (benchmark wrapper uses static fit)
LAMBDA_MODE="${LAMBDA_MODE:-mean}"              # mean | max | ewma
LAMBDA_WINDOW_SIZE="${LAMBDA_WINDOW_SIZE:--1}"  # -1 means full-prefix aggregation
ALPHA="${ALPHA:-1.0}"                           # T3 alpha on chunk_energy
BETA="${BETA:-1.0}"                             # T3 beta on chunk_margin
EWMA_ALPHA="${EWMA_ALPHA:-0.9}"                 # EWMA smoothing when LAMBDA_MODE=ewma

if [[ -z "${DSM_CKPT}" ]]; then
  echo "[run_lpb_score_benchmark] Missing DSM_CKPT=/abs/path/to/model.pt" >&2
  exit 1
fi

if [[ ! -f "${DSM_CKPT}" ]]; then
  echo "[run_lpb_score_benchmark] Checkpoint not found: ${DSM_CKPT}" >&2
  exit 1
fi

# Resolve default eval output: <parent-of-DSM-run-dir>/eval/<RUN_NAME>/benchmark.json
DSM_ABS="$(cd "$(dirname "$DSM_CKPT")" && pwd)/$(basename "$DSM_CKPT")"
if [[ -z "${EVAL_BASE}" ]]; then
  EVAL_BASE="$(dirname "$(dirname "$DSM_ABS")")/eval"
fi
if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
fi
if [[ -z "${OUT_JSON}" ]]; then
  OUT_JSON="${EVAL_BASE}/${RUN_NAME}/benchmark.json"
fi
mkdir -p "$(dirname "$OUT_JSON")"

echo "[run_lpb_score_benchmark] dsm_ckpt=${DSM_ABS}"
echo "[run_lpb_score_benchmark] eval_base=${EVAL_BASE}"
echo "[run_lpb_score_benchmark] run_name=${RUN_NAME}"
echo "[run_lpb_score_benchmark] tasks=${TASKS}"
echo "[run_lpb_score_benchmark] lambda_mode=${LAMBDA_MODE} ewma_alpha=${EWMA_ALPHA}"
echo "[run_lpb_score_benchmark] out_json=${OUT_JSON}"

"${PYTHON_BIN}" -m data.utils.benchmark.examples.run_lpb_score \
  --dsm-ckpt "${DSM_ABS}" \
  --fail-root "${FAIL_ROOT}" \
  --success-root "${SUCCESS_ROOT}" \
  --tasks ${TASKS} \
  --save-json "${OUT_JSON}" \
  --device "${DEVICE}" \
  --feature-batch-size "${FEATURE_BATCH_SIZE}" \
  --delta "${DELTA}" \
  --delta-step "${DELTA_STEP}" \
  --lambda-mode "${LAMBDA_MODE}" \
  --lambda-window-size "${LAMBDA_WINDOW_SIZE}" \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --ewma-alpha "${EWMA_ALPHA}"
