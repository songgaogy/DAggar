#!/usr/bin/env bash
# D3-Disc visualization entry point (fit-on-one-task then render mp4 + pdf).
# Run from repo root:
#   bash robosuite/discriminator/d3disc/scripts/visualize_d3.sh
# Overrides via env vars, e.g.:
#   TASK=PickPlaceBread NUM_TRAJS=4 OMEGA=0.5 \
#     bash robosuite/discriminator/d3disc/scripts/visualize_d3.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/utils/fail_rollout}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/utils/success_rollout}"
TASK="${TASK:-PickPlaceBread}"
NUM_TRAJS="${NUM_TRAJS:-4}"
SEED="${SEED:-0}"
FPS="${FPS:-20}"
BORDER_THICKNESS="${BORDER_THICKNESS:-10}"

# --------------------------------------
# Per-task data caps.
SUCC_NUM="${SUCC_NUM:-200}"
FAIL_NUM="${FAIL_NUM:-100}"

# Optional trained dynamics predictor (from train_d3_dynamics.sh).
DYN_CKPT="checkpoints/d3disc/dynamics/d3dyn_20260422_034143/d3_dynamics.pt"

# D3 hyperparameters.
OMEGA="${OMEGA:-0.5}"
K="${K:-1}"
BETA="${BETA:-auto}"
KAPPA="${KAPPA:-auto}"
SIGMA_SQ="${SIGMA_SQ:-0.5}"
# --------------------------------------

RUN_NAME="${RUN_NAME:-viz_${TASK}_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/d3disc/viz/${RUN_NAME}}"
PDF_NAME="${PDF_NAME:-d3_scores.pdf}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

POLICY_CKPT="${POLICY_CKPT:-${REPO_ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
if [[ ! -f "${POLICY_CKPT}" ]]; then
    echo "[d3_disc][viz] ERROR: POLICY_CKPT not found: ${POLICY_CKPT}" >&2
    exit 1
fi

CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/data/.lpb_score_cache}"
DEVICE="${DEVICE:-cuda}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-256}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"

DELTA="${DELTA:-10.0}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-8192}"
LAMBDA_MODE="${LAMBDA_MODE:-mean}"
LAMBDA_WINDOW_SIZE="${LAMBDA_WINDOW_SIZE:--1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"

EXTRA_ARGS=()
if [[ -n "${SUCC_NUM}" && "${SUCC_NUM}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${SUCC_NUM}")
fi
if [[ -n "${FAIL_NUM}" && "${FAIL_NUM}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${FAIL_NUM}")
fi
if [[ "${QUIET_FIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--quiet-fit)
fi
if [[ -n "${DYN_CKPT}" ]]; then
    if [[ ! -f "${DYN_CKPT}" ]]; then
        echo "[d3_disc][viz] ERROR: DYN_CKPT not found: ${DYN_CKPT}" >&2
        exit 1
    fi
    EXTRA_ARGS+=(--dynamics-ckpt "${DYN_CKPT}")
fi

"${PYTHON_BIN}" -m robosuite.discriminator.d3disc.visualize \
    --policy-ckpt          "${POLICY_CKPT}" \
    --cache-root           "${CACHE_ROOT}" \
    --fail-root            "${FAIL_ROOT}" \
    --success-root         "${SUCCESS_ROOT}" \
    --task                 "${TASK}" \
    --num-trajs            "${NUM_TRAJS}" \
    --out-dir              "${OUT_DIR}" \
    --pdf-name             "${PDF_NAME}" \
    --seed                 "${SEED}" \
    --fps                  "${FPS}" \
    --border-thickness     "${BORDER_THICKNESS}" \
    --device               "${DEVICE}" \
    --encoder-batch-size   "${ENCODER_BATCH_SIZE}" \
    --image-size           "${IMAGE_SIZE}" \
    --camera-name          "${CAMERA_NAME}" \
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
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[d3_disc][viz] wrote to ${OUT_DIR}"
