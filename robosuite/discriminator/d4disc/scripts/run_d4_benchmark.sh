#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"
PREPROCESSED_CACHE_ROOT="${PREPROCESSED_CACHE_ROOT:-${REPO_ROOT}/data/.lpb_score_preprocessed_cache}"

D4_CKPT="checkpoints/d4disc/dynamics/d4_lpb_degraded/d4_dynamics_Aep0020.pt"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/utils/fail_rollout}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/utils/success_rollout}"
TASKS="PickPlaceBread PickPlaceCereal PickPlaceMilk PandaPickPlaceCan"

SUCC_NUM="${SUCC_NUM:-200}"
FAIL_NUM="${FAIL_NUM:-100}"

OMEGA="${OMEGA:-0}"
SIGMA_SQ="${SIGMA_SQ:-}"
HORIZON="${HORIZON:-1}"
DELTA="${DELTA:-10.0}"
LAMBDA_MODE="${LAMBDA_MODE:-mean}"
LAMBDA_WINDOW_SIZE="${LAMBDA_WINDOW_SIZE:--1}"
# Per-step score form.
#   "knn"  (default) : LPB-parity. Builds [obs_proj, proprio_proj,
#                      mean action_proj] (L2-normalized) from the trained
#                      D4 predictor and scores via a per-task KNN bank.
#                      Ignores OMEGA / SIGMA_SQ / SCORING_BATCH_SIZE for
#                      scoring; honors DELTA / LAMBDA_MODE / LAMBDA_WINDOW_SIZE
#                      identically to LPB. This is the evaluation default
#                      until Phase-B + omega>0 empirically beats it.
#   "rel"            : motion-baselined residual NLL; requires Phase-B.
#   "abs"            : raw residual NLL; d4disc_0423.md §8.2 verbatim.
SCORE_MODE="${SCORE_MODE:-knn}"
# Trajectory aggregator: D4 (pos-branch, omega=0) per-step gap is small,
# so `max` is dominated by success-trajectory outliers; `mean` recovers
# ~25 AUROC points pooled on the pick-place tasks. Override with `max` to
# reproduce LPB-style behavior when per-step signal is strong.
TRAJ_SCORE_AGGREGATOR="${TRAJ_SCORE_AGGREGATOR:-mean}"
TRAJ_SCORE_TOPK="${TRAJ_SCORE_TOPK:-20}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
PER_TASK_CALIB="${PER_TASK_CALIB:-1}"
SCORING_BATCH_SIZE="${SCORING_BATCH_SIZE:-512}"

DEVICE="${DEVICE:-cuda:1}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-256}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
SEED="${SEED:-0}"

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/d4disc/eval/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${SUCC_NUM}" && "${SUCC_NUM}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${SUCC_NUM}")
fi
if [[ -n "${FAIL_NUM}" && "${FAIL_NUM}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${FAIL_NUM}")
fi
if [[ -n "${SIGMA_SQ}" ]]; then
    EXTRA_ARGS+=(--sigma-sq "${SIGMA_SQ}")
fi
if [[ "${PER_TASK_CALIB}" == "1" ]]; then
    EXTRA_ARGS+=(--per-task-calibration)
else
    EXTRA_ARGS+=(--shared-calibration)
fi
if [[ "${QUIET_FIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--quiet-fit)
fi

"${PYTHON_BIN}" -m data.utils.benchmark.examples.run_d4 \
    --d4-ckpt                "${D4_CKPT}" \
    --preprocessed-cache-root "${PREPROCESSED_CACHE_ROOT}" \
    --fail-root              "${FAIL_ROOT}" \
    --success-root           "${SUCCESS_ROOT}" \
    --tasks                  ${TASKS} \
    --save-json              "${SAVE_JSON}" \
    --device                 "${DEVICE}" \
    --encoder-batch-size     "${ENCODER_BATCH_SIZE}" \
    --image-size             "${IMAGE_SIZE}" \
    --omega                  "${OMEGA}" \
    --horizon                "${HORIZON}" \
    --delta                  "${DELTA}" \
    --lambda-mode            "${LAMBDA_MODE}" \
    --lambda-window-size     "${LAMBDA_WINDOW_SIZE}" \
    --score-mode             "${SCORE_MODE}" \
    --traj-score-aggregator  "${TRAJ_SCORE_AGGREGATOR}" \
    --traj-score-topk        "${TRAJ_SCORE_TOPK}" \
    --calib-fraction         "${CALIB_FRACTION}" \
    --scoring-batch-size     "${SCORING_BATCH_SIZE}" \
    --seed                   "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[d4_disc] wrote ${SAVE_JSON}"
