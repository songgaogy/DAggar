#!/usr/bin/env bash
# BCE discriminator visualization for robosuite sim data (MP4 + PDF).
#
# Renders per-trajectory MP4 (HUD + red border on predicted-failure frames) and a
# multi-page PDF summary using a fitted BCE discriminator. Success rollouts are
# scored only on their pre-done (is_success==False) prefix.
#
# Required env:
#   MODEL_CKPT=/path/to/checkpoints/model_<epoch>.pth (optional; hardcoded below)
#     bash robosuite/discriminator/dyn_disc/scripts/visualize_bce_robosuite.sh
#
# Layout assumptions:
#   DATA_ROOT/<task>/fail_rollout-labeled      (BCE failure bank)
#   DATA_ROOT/<task>/fail_rollout-val-labeled  (benchmark eval failures)
#   DATA_ROOT/<task>/success_rollout           (BCE train positives + calib)
#   DATA_ROOT/<task>/success_rollout-val       (benchmark eval success)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

# ====================================================================== #
# Important parameters: set directly (no ${VAR:-default} indirection).    #
# ====================================================================== #
MODEL_CKPT="checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_10.pth"
LOAD_CKPT="checkpoints/dyn_disc/bce_eval_robosuite/run_20260619_222530_PickPlaceCereal/checkpoints/bce_head.pth"
TASK="PickPlaceCereal"
NUM_TRAJS=5             # per split: success AND fail each use NUM_TRAJS (SPLIT=both)

FAIL_SPLIT="fail_rollout-val-labeled"        # eval failures
SUCCESS_SPLIT="success_rollout-val"          # eval success (test only)
SUCCESS_TRAIN_SPLIT="success_rollout"        # BCE train positives + calib
FAIL_TRAIN_SPLIT="fail_rollout-labeled"      # GT-labeled failure bank

# Feature space + head knobs (match run_bce_robosuite_benchmark.sh).
FEATURE_SOURCE="transformer"
TRANSFORMER_LAYER=1
HEAD_HIDDEN=256
HEAD_LAYERS=2
FAIL_BANK_PER_TASK=25
MAX_EXPERT_OTHER_RATIO=1.0

# Per-task eval caps.
MAX_FAIL_PER_TASK=100
MAX_SUCCESS_PER_TASK=100
# ====================================================================== #

# Output directory.
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/bce_viz_robosuite/${TASK}-${TIMESTAMP}}"
PDF_NAME="${PDF_NAME:-bce_scores.pdf}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

# Peripheral runtime knobs (still overridable via env).
SEED="${SEED:-0}"
FPS="${FPS:-20}"
BORDER_THICKNESS="${BORDER_THICKNESS:-10}"
DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
VISUAL_WEIGHT="${VISUAL_WEIGHT:-1.0}"
PROPRIO_WEIGHT="${PROPRIO_WEIGHT:-1.0}"
ACTION_WEIGHT="${ACTION_WEIGHT:-4.0}"
DELTA="${DELTA:-10.0}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"

EXTRA_ARGS=()
if [[ "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi
if [[ "${QUIET_FIT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--quiet-fit)
fi
if [[ "${NO_DEBUG_SCORE_STATS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-debug-score-stats)
fi
if [[ "${NO_FLIP_VERTICAL:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-flip-vertical)
fi
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    EXTRA_ARGS+=(--proprio-indices ${PROPRIO_INDICES})
fi
if [[ -n "${CAMERA_TO_VIEW:-}" ]]; then
    EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
fi
if [[ -n "${LOAD_CKPT}" ]]; then
    if [[ ! -f "${LOAD_CKPT}" ]]; then
        echo "[bce][viz][robosuite] ERROR: LOAD_CKPT not found: ${LOAD_CKPT}" >&2
        exit 1
    fi
    EXTRA_ARGS+=(--load-ckpt "${LOAD_CKPT}")
fi
if [[ -n "${SAVE_CKPT_DIR:-}" ]]; then
    EXTRA_ARGS+=(--save-ckpt-dir "${SAVE_CKPT_DIR}")
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.visualization.visualize_bce \
    --model-ckpt            "${MODEL_CKPT}" \
    --data-root             "${DATA_ROOT}" \
    --fail-split            "${FAIL_SPLIT}" \
    --success-split         "${SUCCESS_SPLIT}" \
    --success-train-split   "${SUCCESS_TRAIN_SPLIT}" \
    --fail-train-split      "${FAIL_TRAIN_SPLIT}" \
    --task                  "${TASK}" \
    --num-trajs             "${NUM_TRAJS}" \
    --out-dir               "${OUT_DIR}" \
    --pdf-name              "${PDF_NAME}" \
    --seed                  "${SEED}" \
    --fps                   "${FPS}" \
    --border-thickness      "${BORDER_THICKNESS}" \
    --device                "${DEVICE}" \
    --encode-batch-size     "${ENCODE_BATCH_SIZE}" \
    --camera-name           "${CAMERA_NAME}" \
    --visual-weight         "${VISUAL_WEIGHT}" \
    --proprio-weight        "${PROPRIO_WEIGHT}" \
    --action-weight         "${ACTION_WEIGHT}" \
    --delta                 "${DELTA}" \
    --feature-source        "${FEATURE_SOURCE}" \
    --transformer-layer     "${TRANSFORMER_LAYER}" \
    --calib-fraction        "${CALIB_FRACTION}" \
    --head-hidden           "${HEAD_HIDDEN}" \
    --head-layers           "${HEAD_LAYERS}" \
    --epochs                "${EPOCHS}" \
    --lr                    "${LR}" \
    --weight-decay          "${WEIGHT_DECAY}" \
    --batch-size            "${BATCH_SIZE}" \
    --max-expert-other-ratio "${MAX_EXPERT_OTHER_RATIO}" \
    --fail-bank-per-task    "${FAIL_BANK_PER_TASK}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[bce][viz][robosuite] wrote to ${OUT_DIR}"
