#!/usr/bin/env bash
# Evaluate a saved parent or finetuned nnPU checkpoint without fitting a head.
# Finetuned checkpoints also generate the visualization bundle by default, while
# keeping evaluation and visualization artifacts in separate directories.

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
DEVICE="${DEVICE:-cuda:0}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
CAMERA_TO_VIEW="${CAMERA_TO_VIEW:-}"
SEED="${SEED:-0}"
GENERATE_VISUALS="${GENERATE_VISUALS:-True}"
SPLIT="${SPLIT:-both}"
NUM_TRAJS="${NUM_TRAJS:-10}"
CAMERA_NAME="${CAMERA_NAME:-agentview}"
FPS="${FPS:-20}"
BORDER_THICKNESS="${BORDER_THICKNESS:-10}"
OFFLINE_VIDEO_SIZE="${OFFLINE_VIDEO_SIZE:-256}"

if [[ -z "${FINETUNED_CKPT}" || ! -f "${FINETUNED_CKPT}" ]]; then
  echo "[ERROR] FINETUNED_CKPT must point to a parent or finetuned nnPU checkpoint: ${FINETUNED_CKPT:-<unset>}" >&2
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
OUT_DIR="${OUT_DIR:-${RUN_DIR}/evaluation/val-seed${SEED}}"
VIS_OUT_DIR="${VIS_OUT_DIR:-${RUN_DIR}/visualization/val-seed${SEED}}"
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ -n "${CAMERA_TO_VIEW}" ]]; then
  EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
fi

echo "[eval_disc_finetuned] task=${TASK} seed=${SEED}"
echo "[eval_disc_finetuned] checkpoint=${FINETUNED_CKPT}"
echo "[eval_disc_finetuned] validation=${FAIL_SPLIT},${SUCCESS_SPLIT}"
echo "[eval_disc_finetuned] offline_success=${OFFLINE_EPISODES} (full policy-only pool)"
echo "[eval_disc_finetuned] output=${OUT_DIR} device=${DEVICE}"

"${PY}" -m robosuite.pipeline.offline.src.eval_disc_checkpoint \
  --load-ckpt "${FINETUNED_CKPT}" \
  --model-ckpt "${MODEL_CKPT}" \
  --data-root "${DATA_ROOT}" \
  --offline-episodes "${OFFLINE_EPISODES}" \
  --fail-split "${FAIL_SPLIT}" \
  --success-split "${SUCCESS_SPLIT}" \
  --task "${TASK}" \
  --out-dir "${OUT_DIR}" \
  --device "${DEVICE}" \
  --encode-batch-size "${ENCODE_BATCH_SIZE}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "[eval_disc_finetuned] wrote ${OUT_DIR}/benchmark.json"
echo "[eval_disc_finetuned] wrote ${OUT_DIR}/success_false_alarm.json"
echo "[eval_disc_finetuned] wrote ${OUT_DIR}/offline_success_false_alarm.json"
echo "[eval_disc_finetuned] wrote ${OUT_DIR}/gt_fail_detection.json"

case "${GENERATE_VISUALS,,}" in
  true|1|yes|on)
    echo "[eval_disc_finetuned] generating visualization bundle"
    ROOT_DIR="${ROOT_DIR}" \
    PY="${PY}" \
    TASK="${TASK}" \
    FINETUNED_CKPT="${FINETUNED_CKPT}" \
    MODEL_CKPT="${MODEL_CKPT}" \
    DATA_ROOT="${DATA_ROOT}" \
    OFFLINE_EPISODES="${OFFLINE_EPISODES}" \
    FAIL_SPLIT="${FAIL_SPLIT}" \
    SUCCESS_SPLIT="${SUCCESS_SPLIT}" \
    SPLIT="${SPLIT}" \
    NUM_TRAJS="${NUM_TRAJS}" \
    SEED="${SEED}" \
    DEVICE="${DEVICE}" \
    ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE}" \
    CAMERA_NAME="${CAMERA_NAME}" \
    CAMERA_TO_VIEW="${CAMERA_TO_VIEW}" \
    FPS="${FPS}" \
    BORDER_THICKNESS="${BORDER_THICKNESS}" \
    OFFLINE_VIDEO_SIZE="${OFFLINE_VIDEO_SIZE}" \
    OUT_DIR="${VIS_OUT_DIR}" \
      bash "${ROOT_DIR}/robosuite/pipeline/offline/scripts/vis_disc_finetuned.sh"
    echo "[eval_disc_finetuned] visualization bundle=${VIS_OUT_DIR}"
    echo "[eval_disc_finetuned] wrote ${VIS_OUT_DIR}/finetuned_scores.pdf"
    echo "[eval_disc_finetuned] wrote ${VIS_OUT_DIR}/finetuned_scores_offline.pdf"
    echo "[eval_disc_finetuned] wrote ${VIS_OUT_DIR}/finetuned_scores_offline-success.pdf"
    echo "[eval_disc_finetuned] wrote sampled MP4 files under ${VIS_OUT_DIR}/videos"
    ;;
  false|0|no|off)
    echo "[eval_disc_finetuned] visualization bundle disabled (GENERATE_VISUALS=${GENERATE_VISUALS})"
    ;;
  *)
    echo "[ERROR] GENERATE_VISUALS must be True or False, got: ${GENERATE_VISUALS}" >&2
    exit 2
    ;;
esac
