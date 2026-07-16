#!/usr/bin/env bash
# Render fixed validation success/failure trajectories with one finetuned nnPU
# checkpoint. Outputs one MP4 per trajectory and separate eval/offline PDFs,
# including policy-only successful offline episodes.

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
OFFLINE_EPISODES="${OFFLINE_EPISODES:-${DATA_ROOT}/${TASK}/offline_data/offline_episodes.pt}"
FAIL_SPLIT="${FAIL_SPLIT:-fail_rollout-val-labeled}"
SUCCESS_SPLIT="${SUCCESS_SPLIT:-success_rollout-val}"
SPLIT="${SPLIT:-both}"
NUM_TRAJS="${NUM_TRAJS:-10}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda:0}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
CAMERA_TO_VIEW="${CAMERA_TO_VIEW:-}"
FPS="${FPS:-20}"
BORDER_THICKNESS="${BORDER_THICKNESS:-10}"
OFFLINE_VIDEO_SIZE="${OFFLINE_VIDEO_SIZE:-256}"


if [[ -z "${FINETUNED_CKPT}" || ! -f "${FINETUNED_CKPT}" ]]; then
  echo "[ERROR] FINETUNED_CKPT must point to pu_bce_head_finetuned.pth: ${FINETUNED_CKPT:-<unset>}" >&2
  exit 1
fi
if [[ -z "${MODEL_CKPT}" || ! -f "${MODEL_CKPT}" ]]; then
  echo "[ERROR] MODEL_CKPT must point to the frozen dynamics encoder: ${MODEL_CKPT:-<unset>}" >&2
  exit 1
fi
if [[ -z "${OFFLINE_EPISODES}" || ! -f "${OFFLINE_EPISODES}" ]]; then
  echo "[ERROR] OFFLINE_EPISODES must point to offline_episodes.pt: ${OFFLINE_EPISODES:-<unset>}" >&2
  exit 1
fi

RUN_DIR="$(cd "$(dirname "${FINETUNED_CKPT}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/visualization/val-seed${SEED}}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUT_DIR}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

EXTRA_ARGS=()
if [[ -n "${CAMERA_TO_VIEW}" ]]; then
  EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
fi

echo "[vis_disc_finetuned] task=${TASK} split=${SPLIT} seed=${SEED} num_trajs=${NUM_TRAJS}"
echo "[vis_disc_finetuned] checkpoint=${FINETUNED_CKPT}"
echo "[vis_disc_finetuned] validation=${FAIL_SPLIT},${SUCCESS_SPLIT}"
echo "[vis_disc_finetuned] offline_episodes=${OFFLINE_EPISODES} pools=offline,offline-success"
echo "[vis_disc_finetuned] output=${OUT_DIR} device=${DEVICE}"

"${PY}" -m robosuite.pipeline.offline.src.visualize_disc_finetuned \
  --model-ckpt "${MODEL_CKPT}" \
  --load-ckpt "${FINETUNED_CKPT}" \
  --data-root "${DATA_ROOT}" \
  --fail-split "${FAIL_SPLIT}" \
  --success-split "${SUCCESS_SPLIT}" \
  --offline-episodes "${OFFLINE_EPISODES}" \
  --offline-video-size "${OFFLINE_VIDEO_SIZE}" \
  --split "${SPLIT}" \
  --task "${TASK}" \
  --num-trajs "${NUM_TRAJS}" \
  --out-dir "${OUT_DIR}" \
  --pdf-name "finetuned_scores.pdf" \
  --device "${DEVICE}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE}" \
  --camera-name "${CAMERA_NAME}" \
  --feature-source transformer \
  --transformer-layer 1 \
  --seed "${SEED}" \
  --fps "${FPS}" \
  --border-thickness "${BORDER_THICKNESS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "[vis_disc_finetuned] wrote MP4 files and PDFs: ${OUT_DIR}/finetuned_scores.pdf, ${OUT_DIR}/finetuned_scores_offline.pdf, ${OUT_DIR}/finetuned_scores_offline-success.pdf"

echo "[vis_disc_finetuned] checking GT-fail detection on offline episodes"
"${PY}" -m robosuite.pipeline.offline.discriminator.test_finetuned \
  --load-ckpt "${FINETUNED_CKPT}" \
  --model-ckpt "${MODEL_CKPT}" \
  --offline-episodes "${OFFLINE_EPISODES}" \
  --task "${TASK}" \
  --device "${DEVICE}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE}" \
  --out-dir "${OUT_DIR}" \
  --report-name "gt_fail_detection.json" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

echo "[vis_disc_finetuned] wrote ${OUT_DIR}/gt_fail_detection.json"
