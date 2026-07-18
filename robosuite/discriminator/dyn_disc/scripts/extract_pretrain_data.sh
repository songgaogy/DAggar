#!/usr/bin/env bash
# Extract frozen dynamics latents used to warm-start offline nnPU finetuning.
# This launcher only extracts data; it does not train or evaluate a model.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
TASK="${TASK:-PickPlaceCereal}"
NNPU_CKPT="${NNPU_CKPT:-${REPO_ROOT}/checkpoints/dyn_disc/pu_bce_eval_robosuite-chunk_v2/run_20260717_121451_PickPlaceCereal/checkpoints/pu_bce_head.pth}"
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout}"
FAILURE_SPLIT="${FAILURE_SPLIT:-fail_rollout}"
SUCCESS_CAP="${SUCCESS_CAP:-50}"
FAILURE_CAP="${FAILURE_CAP:-50}"
SEED="${SEED:-0}"
CALIBRATION_FRACTION="${CALIBRATION_FRACTION:-0.2}"
DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/${TASK}/discriminator-pretrain-quadratic-c2-l1e2-v2}"
OVERWRITE="${OVERWRITE:-False}"

if [[ ! -f "${NNPU_CKPT}" ]]; then
    echo "[ERROR] NNPU_CKPT does not exist: ${NNPU_CKPT}" >&2
    exit 2
fi
if [[ "${DEVICE}" != cuda* ]]; then
    echo "[ERROR] DEVICE must be a CUDA device, got: ${DEVICE}" >&2
    exit 2
fi

EXTRA_ARGS=()
case "${OVERWRITE,,}" in
    true|1|yes|on)
        EXTRA_ARGS+=(--overwrite)
        ;;
    false|0|no|off)
        ;;
    *)
        echo "[ERROR] OVERWRITE must be True or False, got: ${OVERWRITE}" >&2
        exit 2
        ;;
esac

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.utils.extract_data \
    --checkpoint "${NNPU_CKPT}" \
    --task "${TASK}" \
    --data-root "${DATA_ROOT}" \
    --success-split "${SUCCESS_SPLIT}" \
    --failure-split "${FAILURE_SPLIT}" \
    --success-cap "${SUCCESS_CAP}" \
    --failure-cap "${FAILURE_CAP}" \
    --seed "${SEED}" \
    --calibration-fraction "${CALIBRATION_FRACTION}" \
    --device "${DEVICE}" \
    --encode-batch-size "${ENCODE_BATCH_SIZE}" \
    --output-dir "${OUTPUT_DIR}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[extract_data] manifest: ${OUTPUT_DIR}/manifest.json"
