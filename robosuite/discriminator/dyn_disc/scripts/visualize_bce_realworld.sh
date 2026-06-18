#!/usr/bin/env bash
# BCE discriminator visualization for real-world Agilex data.
#
# Renders per-failure-trajectory MP4 (with HUD + red border on predicted-failure
# frames) and a multi-page PDF summary using a fitted BCE discriminator.
#
# Required env:
#   MODEL_CKPT=/path/to/checkpoints/model_<epoch>.pth \
#     bash robosuite/discriminator/dyn_disc/scripts/visualize_bce_realworld.sh
#
# Optional env (commonly overridden):
#   TASK, NUM_TRAJS, CAMERA_NAME, CAMERA_TO_VIEW, LOAD_CKPT (skip fit),
#   FAIL_BANK_PER_TASK, MAX_FAIL_PER_TASK, MAX_SUCCESS_PER_TASK
#
# Mirrors `run_bce_real_world_benchmark.sh`: bank pool is drawn from the same
# FAIL_ROOT, with disjointness enforced by filtering eval video_ids.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/agilex/failure_annotations/out_by_task}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/agilex}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/data/.agilex_train_cache}"

# -------------------------------------------------
TASK="candy_in_plate"
NUM_TRAJS=3
SEED="${SEED:-0}"
FPS="${FPS:-20}"
BORDER_THICKNESS="${BORDER_THICKNESS:-10}"
# -------------------------------------------------


MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-50}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-50}"

RUN_NAME="${RUN_NAME:-viz_bce_${TASK}}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/bce_viz_realworld/${RUN_NAME}-${TIMESTAMP}}"
PDF_NAME="${PDF_NAME:-bce_v2_scores.pdf}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/dynamics/agilex_train-20260429_003751/checkpoints/model_49.pth}"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[bce][viz][realworld] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    echo "                 Set MODEL_CKPT=/abs/path/to/checkpoints/model_<epoch>.pth." >&2
    exit 1
fi

# Right-arm Agilex defaults: qpos[:, 7:14], action[:, 7:14].
PROPRIO_FIELD="${PROPRIO_FIELD:-qpos}"
PROPRIO_START="${PROPRIO_START:-7}"
PROPRIO_STOP="${PROPRIO_STOP:-14}"
ACTION_START="${ACTION_START:-7}"
ACTION_STOP="${ACTION_STOP:-14}"

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
CAMERA_NAME="${CAMERA_NAME:-cam_high}"
CAMERA_TO_VIEW="${CAMERA_TO_VIEW:-cam_high:agentview}"
VISUAL_WEIGHT="${VISUAL_WEIGHT:-1.0}"
PROPRIO_WEIGHT="${PROPRIO_WEIGHT:-2.0}"
ACTION_WEIGHT="${ACTION_WEIGHT:-1.0}"
DELTA="${DELTA:-10.0}"
FEATURE_SOURCE="${FEATURE_SOURCE:-transformer}"
TRANSFORMER_LAYER="${TRANSFORMER_LAYER:-1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"

# ------------------------------------------------------------------------
# BCE head + optim knobs (match run_bce_real_world_benchmark.sh defaults).
FAIL_BANK_PER_TASK="${FAIL_BANK_PER_TASK:-25}"
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_LAYERS="${HEAD_LAYERS:-2}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_EXPERT_OTHER_RATIO="${MAX_EXPERT_OTHER_RATIO:-1.0}"
# ------------------------------------------------------------------------

EXTRA_ARGS=()
if [[ "${USE_CACHE:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--cache-root "${CACHE_ROOT}")
fi
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
if [[ -n "${CAMERA_TO_VIEW:-}" ]]; then
    EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
fi
if [[ -n "${LOAD_CKPT:-}" ]]; then
    if [[ ! -f "${LOAD_CKPT}" ]]; then
        echo "[bce][viz][realworld] ERROR: LOAD_CKPT not found: ${LOAD_CKPT}" >&2
        exit 1
    fi
    EXTRA_ARGS+=(--load-ckpt "${LOAD_CKPT}")
fi
if [[ -n "${SAVE_CKPT_DIR:-}" ]]; then
    EXTRA_ARGS+=(--save-ckpt-dir "${SAVE_CKPT_DIR}")
fi

SPLIT="${SPLIT:-fail_rollout}"

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.visualization.visualize_bce \
    --kind                  realworld \
    --split                 "${SPLIT}" \
    --model-ckpt            "${MODEL_CKPT}" \
    --fail-root             "${FAIL_ROOT}" \
    --success-root          "${SUCCESS_ROOT}" \
    --task                  "${TASK}" \
    --num-trajs             "${NUM_TRAJS}" \
    --out-dir               "${OUT_DIR}" \
    --pdf-name              "${PDF_NAME}" \
    --seed                  "${SEED}" \
    --fps                   "${FPS}" \
    --border-thickness      "${BORDER_THICKNESS}" \
    --proprio-field         "${PROPRIO_FIELD}" \
    --proprio-start         "${PROPRIO_START}" \
    --proprio-stop          "${PROPRIO_STOP}" \
    --action-start          "${ACTION_START}" \
    --action-stop           "${ACTION_STOP}" \
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

echo "[bce][viz][realworld] wrote to ${OUT_DIR}"
