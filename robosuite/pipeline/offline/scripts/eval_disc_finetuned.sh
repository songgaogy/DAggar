#!/usr/bin/env bash
# Evaluate a saved parent or finetuned nnPU checkpoint without fitting a head.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONFAULTHANDLER=1

# ---------------------------------------------------------------
TASK="${TASK:-PickPlaceCereal}"
FINETUNED_CKPT="${FINETUNED_CKPT:-}"
MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_10.pth}"
# ---------------------------------------------------------------

DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
FAIL_SPLIT="${FAIL_SPLIT:-fail_rollout-val-labeled}"
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout-val}"
DEVICE="${DEVICE:-cuda:0}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
CAMERA_TO_VIEW="${CAMERA_TO_VIEW:-}"
SEED="${SEED:-0}"

if [[ -z "${FINETUNED_CKPT}" || ! -f "${FINETUNED_CKPT}" ]]; then
  echo "[ERROR] FINETUNED_CKPT must point to a parent or finetuned nnPU checkpoint: ${FINETUNED_CKPT:-<unset>}" >&2
  exit 1
fi
if [[ -z "${MODEL_CKPT}" || ! -f "${MODEL_CKPT}" ]]; then
  echo "[ERROR] MODEL_CKPT must point to the frozen dynamics encoder: ${MODEL_CKPT:-<unset>}" >&2
  exit 1
fi

RUN_DIR="$(cd "$(dirname "${FINETUNED_CKPT}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/evaluation/val-seed${SEED}}"
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${CAMERA_TO_VIEW}" ]]; then
  EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
fi

echo "[eval_disc_finetuned] task=${TASK} seed=${SEED}"
echo "[eval_disc_finetuned] checkpoint=${FINETUNED_CKPT}"
echo "[eval_disc_finetuned] validation=${FAIL_SPLIT},${SUCCESS_SPLIT}"
echo "[eval_disc_finetuned] output=${OUT_DIR} device=${DEVICE}"

"${PY}" -m robosuite.pipeline.offline.src.eval_disc_checkpoint \
  --load-ckpt "${FINETUNED_CKPT}" \
  --model-ckpt "${MODEL_CKPT}" \
  --data-root "${DATA_ROOT}" \
  --fail-split "${FAIL_SPLIT}" \
  --success-split "${SUCCESS_SPLIT}" \
  --task "${TASK}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "[eval_disc_finetuned] wrote ${OUT_DIR}/benchmark.json and ${OUT_DIR}/success_false_alarm.json"
