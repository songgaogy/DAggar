#!/usr/bin/env bash
# logpZO benchmark entry point for the real-world Agilex dataset.
# Run from repo root:
#   bash robosuite/discriminator/logpZO/scripts/run_logpZO_real_world_benchmark.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/agilex/failure_annotations/out_by_task}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/agilex}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/data/.agilex_train_cache}"
TASKS="${TASKS:-candy_in_plate duck_in_bowl Micky_in_box sausage_in_pot}"

MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-25}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-100}"

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/logpZO/real_world_eval/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

ACTION_START="${ACTION_START:-7}"
ACTION_STOP="${ACTION_STOP:-13}"
PROPRIO_FIELD="${PROPRIO_FIELD:-qpos}"
PROPRIO_START="${PROPRIO_START:-7}"
PROPRIO_STOP="${PROPRIO_STOP:-14}"

DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-64}"
CAMERA_NAME="${CAMERA_NAME:-cam_high}"

NUM_LAYERS="${NUM_LAYERS:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
SCALE_CLAMP="${SCALE_CLAMP:-3.0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
ALPHA="${ALPHA:-0.1}"
CALIB_FRACTION="${CALIB_FRACTION:-0.2}"
SEED="${SEED:-0}"

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
if [[ "${USE_CACHE:-1}" == "1" && -d "${CACHE_ROOT}" ]]; then
    EXTRA_ARGS+=(--cache-root "${CACHE_ROOT}")
fi

"${PYTHON_BIN}" -m benchmark.real_world.examples.run_logpZO \
    --fail-root            "${FAIL_ROOT}" \
    --success-root         "${SUCCESS_ROOT}" \
    --tasks                ${TASKS} \
    --save-json            "${SAVE_JSON}" \
    --proprio-field        "${PROPRIO_FIELD}" \
    --proprio-start        "${PROPRIO_START}" \
    --proprio-stop         "${PROPRIO_STOP}" \
    --action-start         "${ACTION_START}" \
    --action-stop          "${ACTION_STOP}" \
    --device               "${DEVICE}" \
    --image-size           "${IMAGE_SIZE}" \
    --encoder-batch-size   "${ENCODER_BATCH_SIZE}" \
    --camera-name          "${CAMERA_NAME}" \
    --num-layers           "${NUM_LAYERS}" \
    --hidden-dim           "${HIDDEN_DIM}" \
    --scale-clamp          "${SCALE_CLAMP}" \
    --epochs               "${EPOCHS}" \
    --batch-size           "${BATCH_SIZE}" \
    --lr                   "${LR}" \
    --weight-decay         "${WEIGHT_DECAY}" \
    --val-fraction         "${VAL_FRACTION}" \
    --early-stop-patience  "${EARLY_STOP_PATIENCE}" \
    --alpha                "${ALPHA}" \
    --calib-fraction       "${CALIB_FRACTION}" \
    --seed                 "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[real_world][logpZO] wrote ${SAVE_JSON}"
