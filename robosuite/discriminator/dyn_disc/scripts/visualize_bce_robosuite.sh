#!/usr/bin/env bash
# BCE discriminator visualization for robosuite sim data.
#
# Renders per-failure-trajectory MP4 (with HUD + red border on predicted-failure
# frames) and a multi-page PDF summary using a fitted BCE discriminator.
#
# Required env:
#   MODEL_CKPT=/path/to/checkpoints/model_<epoch>.pth \
#     bash robosuite/discriminator/dyn_disc/scripts/visualize_bce_robosuite.sh
#
# Optional env (commonly overridden):
#   TASK, NUM_TRAJS, CAMERA_NAME, LOAD_CKPT (skip fit), FAIL_BANK_PER_TASK,
#   MAX_FAIL_PER_TASK, MAX_SUCCESS_PER_TASK
#
# Layout assumptions (override via env vars):
#   DATA_ROOT/<task>/fail_rollout-labeled      (BCE failure bank)
#   DATA_ROOT/<task>/fail_rollout-val-labeled  (benchmark eval failures)
#   DATA_ROOT/<task>/success_rollout-val       (benchmark eval success)

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
FAIL_SPLIT="${FAIL_SPLIT:-fail_rollout-val-labeled}"
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout-val}"
FAIL_TRAIN_SPLIT="${FAIL_TRAIN_SPLIT:-fail_rollout-labeled}"

# -------------------------------------------------
TASK="${TASK:-PickPlaceMilk}"
SPLIT="${SPLIT:-success_rollout}"     # success_rollout or fail_rollout
NUM_TRAJS="${NUM_TRAJS:-3}"
SEED="${SEED:-0}"
FPS="${FPS:-20}"
BORDER_THICKNESS="${BORDER_THICKNESS:-10}"
# Threshold: visualizer always computes two-class Youden on the sampled task.
# To change the operating point, re-fit with a different CALIB_MODE in
# run_bce_robosuite_benchmark.sh and re-run.
# -------------------------------------------------


MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-100}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-100}"

RUN_NAME="${RUN_NAME:-viz_bce_${TASK}}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/bce_viz_robosuite-${SPLIT}/${RUN_NAME}-${TIMESTAMP}}"
PDF_NAME="${PDF_NAME:-bce_v2_scores.pdf}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/dynamics/train-20260428_210501/checkpoints/model_49.pth}"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[bce][viz][robosuite] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    echo "                 Set MODEL_CKPT=/abs/path/to/checkpoints/model_<epoch>.pth." >&2
    exit 1
fi

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
VISUAL_WEIGHT="${VISUAL_WEIGHT:-1.0}"
PROPRIO_WEIGHT="${PROPRIO_WEIGHT:-1.0}"
ACTION_WEIGHT="${ACTION_WEIGHT:-4.0}"
DELTA="${DELTA:-2.0}"
FEATURE_SOURCE="${FEATURE_SOURCE:-transformer}"
TRANSFORMER_LAYER="${TRANSFORMER_LAYER:-1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"

# ----------------------------------------------------------------------
# BCE head + optim knobs (match run_bce_robosuite_benchmark.sh defaults).
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_LAYERS="${HEAD_LAYERS:-2}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_EXPERT_OTHER_RATIO="${MAX_EXPERT_OTHER_RATIO:-1.0}"
FAIL_BANK_PER_TASK="${FAIL_BANK_PER_TASK:-25}"
# ----------------------------------------------------------------------


EXTRA_ARGS=()
if [[ -n "${MAX_FAIL_PER_TASK}" && "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_SUCCESS_PER_TASK}" && "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
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
if [[ -n "${LOAD_CKPT:-}" ]]; then
    if [[ ! -f "${LOAD_CKPT}" ]]; then
        echo "[bce][viz][robosuite] ERROR: LOAD_CKPT not found: ${LOAD_CKPT}" >&2
        exit 1
    fi
    EXTRA_ARGS+=(--load-ckpt "${LOAD_CKPT}")
fi
if [[ -n "${SAVE_CKPT_DIR:-}" ]]; then
    EXTRA_ARGS+=(--save-ckpt-dir "${SAVE_CKPT_DIR}")
fi

python -m robosuite.discriminator.dyn_disc.visualization.visualize_bce \
    --kind                  robosuite \
    --split                 "${SPLIT}" \
    --model-ckpt            "${MODEL_CKPT}" \
    --data-root             "${DATA_ROOT}" \
    --fail-split            "${FAIL_SPLIT}" \
    --success-split         "${SUCCESS_SPLIT}" \
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
