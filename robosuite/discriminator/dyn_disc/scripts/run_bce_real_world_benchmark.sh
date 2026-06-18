#!/usr/bin/env bash
# BCE-WAM benchmark (GT failure split) for real-world Agilex data.
# Hard constraint: training does NOT run benchmark evaluation. bench.evaluate
# is called by the runner only after fit_on_benchmark returns.
#
# Required env:
#   MODEL_CKPT=/path/to/checkpoints/model_<epoch>.pth \
#     bash robosuite/discriminator/dyn_disc/scripts/run_bce_real_world_benchmark.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/agilex/failure_annotations/out_by_task}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/agilex}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/data/.agilex_train_cache}"
TASKS="${TASKS:-candy_in_plate duck_in_bowl Micky_in_box sausage_in_pot}"

# --------------------------------------------------------
# for evaluation
MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-25}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-50}"
# --------------------------------------------------------

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/bce_eval_realworld/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
SAVE_CKPT_DIR="${SAVE_CKPT_DIR:-${OUT_DIR}/checkpoints}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/dynamics/agilex_train-20260429_003751/checkpoints/model_49.pth}"
if [[ ! -f "${MODEL_CKPT}" ]]; then
    echo "[real_world][bce] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

# Right-arm Agilex defaults: qpos[:, 7:14], action[:, 7:14].
ACTION_START="${ACTION_START:-7}"
ACTION_STOP="${ACTION_STOP:-14}"
PROPRIO_FIELD="${PROPRIO_FIELD:-qpos}"
PROPRIO_START="${PROPRIO_START:-7}"
PROPRIO_STOP="${PROPRIO_STOP:-14}"

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
VISUAL_WEIGHT="${VISUAL_WEIGHT:-1.0}"
PROPRIO_WEIGHT="${PROPRIO_WEIGHT:-2.0}"
ACTION_WEIGHT="${ACTION_WEIGHT:-1.0}"
DELTA="${DELTA:-10.0}"
KNN_CHUNK_SIZE="${KNN_CHUNK_SIZE:-2048}"
# Locks the feature space to transformer layer 1 (matches diagnostic + two-bank).
KNN_FEATURE_SOURCE="${KNN_FEATURE_SOURCE:-transformer}"
KNN_TRANSFORMER_LAYER="${KNN_TRANSFORMER_LAYER:-1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
SEED="${SEED:-0}"

# --------------------------------------------------------
# BCE head + optim knobs.
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_LAYERS="${HEAD_LAYERS:-2}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"

# Balance: MAX_EXPERT_OTHER_RATIO caps |D_e| <= ratio * |D_o|. <=0 disables.
MAX_EXPERT_OTHER_RATIO="${MAX_EXPERT_OTHER_RATIO:-1.0}"

# Failure pool: disjoint from eval failures; each traj split at first_gt_failure_frame.
FAIL_BANK_PER_TASK="${FAIL_BANK_PER_TASK:-25}"

# Per-task threshold calibration. AUROC / AUPRC are computed from continuous
# step_scores, so they are invariant under CALIB_MODE; only F1 / precision /
# recall move with tau.
#   success_percentile : tau = percentile(success-calib failure scores, 100 - DELTA)
#   two_class_youden   : tau = argmax(TPR - FPR) on (success-calib, fail-suffix)
CALIB_MODE="${CALIB_MODE:-two_class_youden}"
# --------------------------------------------------------


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
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    EXTRA_ARGS+=(--proprio-indices ${PROPRIO_INDICES})
fi
if [[ -n "${CAMERA_TO_VIEW:-}" ]]; then
    EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.real_world_bce \
    --model-ckpt           "${MODEL_CKPT}" \
    --fail-root            "${FAIL_ROOT}" \
    --success-root         "${SUCCESS_ROOT}" \
    --tasks                ${TASKS} \
    --save-json            "${SAVE_JSON}" \
    --save-ckpt-dir        "${SAVE_CKPT_DIR}" \
    --proprio-field        "${PROPRIO_FIELD}" \
    --proprio-start        "${PROPRIO_START}" \
    --proprio-stop         "${PROPRIO_STOP}" \
    --action-start         "${ACTION_START}" \
    --action-stop          "${ACTION_STOP}" \
    --device               "${DEVICE}" \
    --encode-batch-size    "${ENCODE_BATCH_SIZE}" \
    --visual-weight        "${VISUAL_WEIGHT}" \
    --proprio-weight       "${PROPRIO_WEIGHT}" \
    --action-weight        "${ACTION_WEIGHT}" \
    --delta                "${DELTA}" \
    --knn-chunk-size       "${KNN_CHUNK_SIZE}" \
    --knn-feature-source   "${KNN_FEATURE_SOURCE}" \
    --knn-transformer-layer "${KNN_TRANSFORMER_LAYER}" \
    --calib-fraction       "${CALIB_FRACTION}" \
    --seed                 "${SEED}" \
    --head-hidden          "${HEAD_HIDDEN}" \
    --head-layers          "${HEAD_LAYERS}" \
    --epochs               "${EPOCHS}" \
    --lr                   "${LR}" \
    --weight-decay         "${WEIGHT_DECAY}" \
    --batch-size           "${BATCH_SIZE}" \
    --max-expert-other-ratio "${MAX_EXPERT_OTHER_RATIO}" \
    --fail-bank-per-task   "${FAIL_BANK_PER_TASK}" \
    --calib-mode           "${CALIB_MODE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[real_world][bce] wrote ${SAVE_JSON}"
