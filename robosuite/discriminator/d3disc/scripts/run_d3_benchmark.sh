#!/usr/bin/env bash
# D3-Disc benchmark entry point (fit + evaluate in one pass).
# Run from repo root:
#   bash robosuite/discriminator/d3disc/scripts/run_d3_benchmark.sh
# Override via env vars, e.g.:
#   OMEGA=0 SUCC_NUM=150 FAIL_NUM=80 TASKS="PickPlaceBread" \
#     bash robosuite/discriminator/d3disc/scripts/run_d3_benchmark.sh
#
# Requires: POLICY_CKPT pointing to a flow_multi policy checkpoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/utils/fail_rollout}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/utils/success_rollout}"
TASKS="${TASKS:-PandaLift PandaPickPlaceCan PandaStack PickPlaceBread PickPlaceCereal PickPlaceMilk}"

# --------------------------------------
# Per-task data caps. User-facing aliases for benchmark --max-*-per-task.
SUCC_NUM="${SUCC_NUM:-200}"
FAIL_NUM="${FAIL_NUM:-100}"

# Empty means use raw flow_multi z_t directly.
DYN_CKPT="checkpoints/d3disc/dynamics/d3dyn_20260422_034143/d3_dynamics.pt"

# D3 hyperparameters.
OMEGA=0.5
K=1
BETA="${BETA:-auto}"
KAPPA="${KAPPA:-auto}"
SIGMA_SQ="${SIGMA_SQ:-0.5}"
# --------------------------------------

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/d3disc/eval/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# Required flow_multi checkpoint.
POLICY_CKPT="${POLICY_CKPT:-${REPO_ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
if [[ ! -f "${POLICY_CKPT}" ]]; then
    echo "[d3_disc] ERROR: POLICY_CKPT not found: ${POLICY_CKPT}" >&2
    exit 1
fi

CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/data/.lpb_score_cache}"
DEVICE="${DEVICE:-cuda}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-256}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"


# Calibration / threshold.
DELTA="${DELTA:-10.0}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-8192}"
LAMBDA_MODE="${LAMBDA_MODE:-mean}"
LAMBDA_WINDOW_SIZE="${LAMBDA_WINDOW_SIZE:--1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"

SEED="${SEED:-0}"
SHARE_BANKS="${SHARE_BANKS:-1}"

EXTRA_ARGS=()
if [[ -n "${SUCC_NUM}" && "${SUCC_NUM}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${SUCC_NUM}")
fi
if [[ -n "${FAIL_NUM}" && "${FAIL_NUM}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${FAIL_NUM}")
fi
if [[ "${SHARE_BANKS}" == "1" ]]; then
    EXTRA_ARGS+=(--share-banks)
else
    EXTRA_ARGS+=(--per-task-banks)
fi
if [[ "${QUIET_FIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--quiet-fit)
fi
if [[ -n "${DYN_CKPT}" ]]; then
    if [[ ! -f "${DYN_CKPT}" ]]; then
        echo "[d3_disc] ERROR: DYN_CKPT not found: ${DYN_CKPT}" >&2
        exit 1
    fi
    EXTRA_ARGS+=(--dynamics-ckpt "${DYN_CKPT}")
fi
if [[ "${NO_NORMALIZE_FEATURE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-normalize-feature)
fi

"${PYTHON_BIN}" -m data.utils.benchmark.examples.run_d3 \
    --policy-ckpt          "${POLICY_CKPT}" \
    --cache-root           "${CACHE_ROOT}" \
    --fail-root            "${FAIL_ROOT}" \
    --success-root         "${SUCCESS_ROOT}" \
    --tasks                ${TASKS} \
    --save-json            "${SAVE_JSON}" \
    --device               "${DEVICE}" \
    --encoder-batch-size   "${ENCODER_BATCH_SIZE}" \
    --image-size           "${IMAGE_SIZE}" \
    --omega                "${OMEGA}" \
    --k                    "${K}" \
    --beta                 "${BETA}" \
    --kappa                "${KAPPA}" \
    --sigma-sq             "${SIGMA_SQ}" \
    --delta                "${DELTA}" \
    --knn-chunk-size       "${KNN_CHUNK_SIZE}" \
    --lambda-mode          "${LAMBDA_MODE}" \
    --lambda-window-size   "${LAMBDA_WINDOW_SIZE}" \
    --calib-fraction       "${CALIB_FRACTION}" \
    --seed                 "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[d3_disc] wrote ${SAVE_JSON}"
