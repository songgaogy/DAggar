#!/usr/bin/env bash
# Train and evaluate the unchanged nnPU head on frozen RPT action-token features.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/ablations/RPT/pretrain/rpt_robosuite-20260903_192142/checkpoint/model_50.pth}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
TASKS="${TASKS:-NutAssemblySquare PickPlaceCereal Stack}"
FAIL_SPLIT="${FAIL_SPLIT:-fail_rollout-val-labeled}"
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout-val}"
SUCCESS_TRAIN_SPLIT="${SUCCESS_TRAIN_SPLIT:-success_rollout}"
FAIL_TRAIN_SPLIT="${FAIL_TRAIN_SPLIT:-fail_rollout}"

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)_${TASKS// /-}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/RPT/nnpu/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
SAVE_CKPT_DIR="${SAVE_CKPT_DIR:-${OUT_DIR}/checkpoints}"
mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
[[ -f "${MODEL_CKPT}" ]] || { echo "RPT checkpoint not found: ${MODEL_CKPT}" >&2; exit 1; }
[[ -d "${DATA_ROOT}" ]] || { echo "Data root not found: ${DATA_ROOT}" >&2; exit 1; }

TASK_ARGS=()
read -r -a TASK_ARGS <<< "${TASKS}"
EXTRA_ARGS=()
for spec in \
    "TRAIN_MAX_SUCCESS_PER_TASK:--train-max-success-per-task" \
    "TRAIN_MAX_FAIL_PER_TASK:--train-max-fail-per-task" \
    "MAX_FAIL_PER_TASK:--max-fail-per-task" \
    "MAX_SUCCESS_PER_TASK:--max-success-per-task"; do
    variable="${spec%%:*}"
    flag="${spec#*:}"
    value="${!variable:-}"
    if [[ -n "${value}" && "${value}" -gt 0 ]]; then
        EXTRA_ARGS+=("${flag}" "${value}")
    fi
done
if [[ "${NO_NN_CORRECTION:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-nn-correction)
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.robosuite_pu_bce \
    --model-ckpt "${MODEL_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --tasks "${TASK_ARGS[@]}" \
    --fail-split "${FAIL_SPLIT}" \
    --success-split "${SUCCESS_SPLIT}" \
    --success-train-split "${SUCCESS_TRAIN_SPLIT}" \
    --fail-train-split "${FAIL_TRAIN_SPLIT}" \
    --save-json "${SAVE_JSON}" \
    --save-ckpt-dir "${SAVE_CKPT_DIR}" \
    --device cuda \
    --encode-batch-size "${ENCODE_BATCH_SIZE:-128}" \
    --delta "${DELTA:-10.0}" \
    --calib-fraction "${CALIB_FRACTION:-0.2}" \
    --seed "${SEED:-0}" \
    --pi-p "${PI_P:-0.5}" \
    --loss-surrogate "${LOSS_SURROGATE:-logistic}" \
    --beta "${BETA:-0.0}" \
    --head-hidden "${HEAD_HIDDEN:-256}" \
    --head-layers "${HEAD_LAYERS:-2}" \
    --epochs "${EPOCHS:-20}" \
    --lr "${LR:-3e-4}" \
    --weight-decay "${WEIGHT_DECAY:-1e-4}" \
    --batch-size "${BATCH_SIZE:-512}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[rpt][nnpu] wrote ${SAVE_JSON}"
