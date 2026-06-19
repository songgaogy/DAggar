#!/usr/bin/env bash
# nnPU (PU-BCE) discriminator visualization for robosuite sim data.
#
# Renders per-trajectory MP4 (with HUD + red border on predicted-failure frames)
# and a multi-page PDF summary using a fitted PU-BCE discriminator. Threshold is
# the detector's own per-task success_percentile tau (no GT timing / Youden).
#
# Required env:
#   MODEL_CKPT=/path/to/checkpoints/model_<epoch>.pth \
#     bash robosuite/discriminator/dyn_disc/scripts/visualize_pu_bce_robosuite.sh
#
# Optional env: TASK, SPLIT, NUM_TRAJS, CAMERA_NAME, LOAD_CKPT (skip fit),
#   UNLABELED_PER_TASK, PI_P, MAX_FAIL_PER_TASK, MAX_SUCCESS_PER_TASK

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
FAIL_SPLIT="${FAIL_SPLIT:-fail_rollout-val-labeled}"
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout-val}"
FAIL_TRAIN_SPLIT="${FAIL_TRAIN_SPLIT:-fail_rollout-labeled}"

TASK="${TASK:-PickPlaceMilk}"
SPLIT="${SPLIT:-fail_rollout}"     # success_rollout or fail_rollout
NUM_TRAJS="${NUM_TRAJS:-3}"
SEED="${SEED:-0}"
FPS="${FPS:-20}"
BORDER_THICKNESS="${BORDER_THICKNESS:-10}"

MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-100}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-100}"

RUN_NAME="${RUN_NAME:-viz_pu_bce_${TASK}}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/pu_bce_viz_robosuite-${SPLIT}/${RUN_NAME}-${TIMESTAMP}}"
PDF_NAME="${PDF_NAME:-pu_bce_scores.pdf}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_50.pth}"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[pu_bce][viz][robosuite] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
DELTA="${DELTA:-10.0}"
FEATURE_SOURCE="${FEATURE_SOURCE:-transformer}"
TRANSFORMER_LAYER="${TRANSFORMER_LAYER:-1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"

PI_P="${PI_P:-0.5}"
LOSS_SURROGATE="${LOSS_SURROGATE:-sigmoid}"
BETA="${BETA:-0.0}"
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_LAYERS="${HEAD_LAYERS:-2}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UNLABELED_PER_TASK="${UNLABELED_PER_TASK:-25}"

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
if [[ "${NO_NN_CORRECTION:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-nn-correction)
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
        echo "[pu_bce][viz][robosuite] ERROR: LOAD_CKPT not found: ${LOAD_CKPT}" >&2
        exit 1
    fi
    EXTRA_ARGS+=(--load-ckpt "${LOAD_CKPT}")
fi
if [[ -n "${SAVE_CKPT_DIR:-}" ]]; then
    EXTRA_ARGS+=(--save-ckpt-dir "${SAVE_CKPT_DIR}")
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.visualization.visualize_pu_bce \
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
    --delta                 "${DELTA}" \
    --feature-source        "${FEATURE_SOURCE}" \
    --transformer-layer     "${TRANSFORMER_LAYER}" \
    --calib-fraction        "${CALIB_FRACTION}" \
    --pi-p                  "${PI_P}" \
    --loss-surrogate        "${LOSS_SURROGATE}" \
    --beta                  "${BETA}" \
    --head-hidden           "${HEAD_HIDDEN}" \
    --head-layers           "${HEAD_LAYERS}" \
    --epochs                "${EPOCHS}" \
    --lr                    "${LR}" \
    --weight-decay          "${WEIGHT_DECAY}" \
    --batch-size            "${BATCH_SIZE}" \
    --unlabeled-per-task    "${UNLABELED_PER_TASK}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[pu_bce][viz][robosuite] wrote to ${OUT_DIR}"
