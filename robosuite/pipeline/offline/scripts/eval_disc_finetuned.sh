#!/usr/bin/env bash
# Evaluate a saved parent or finetuned nnPU checkpoint without fitting a head.
# Finetuned checkpoints also generate the visualization bundle by default, while
# keeping evaluation and visualization artifacts in separate directories.
# Set RUN_EVAL=False to regenerate only the sampled visualization bundle.

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
DELTA="${DELTA:-5.0}"
CAMERA_TO_VIEW="${CAMERA_TO_VIEW:-}"
SEED="${SEED:-0}"
RUN_EVAL="${RUN_EVAL:-True}"
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
OUT_DIR="${OUT_DIR:-${RUN_DIR}/discriminator/eval-seed${SEED}}"
VIS_OUT_DIR="${VIS_OUT_DIR:-${RUN_DIR}/discriminator/vis-seed${SEED}}"

EXTRA_ARGS=()
if [[ -n "${CAMERA_TO_VIEW}" ]]; then
  EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
fi

_parse_bool() {
  local name="$1"
  local value="$2"
  case "${value,,}" in
    true|1|yes|on) echo true ;;
    false|0|no|off) echo false ;;
    *)
      echo "[ERROR] ${name} must be True or False, got: ${value}" >&2
      exit 2
      ;;
  esac
}

RUN_EVAL_FLAG="$(_parse_bool RUN_EVAL "${RUN_EVAL}")"
GENERATE_VISUALS_FLAG="$(_parse_bool GENERATE_VISUALS "${GENERATE_VISUALS}")"

if [[ "${RUN_EVAL_FLAG}" != "true" && "${GENERATE_VISUALS_FLAG}" != "true" ]]; then
  echo "[ERROR] At least one of RUN_EVAL or GENERATE_VISUALS must be True" >&2
  exit 2
fi

if [[ "${RUN_EVAL_FLAG}" == "true" ]]; then
  mkdir -p "${OUT_DIR}"
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
    --delta "${DELTA}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

  echo "[eval_disc_finetuned] wrote ${OUT_DIR}/benchmark.json"
  echo "[eval_disc_finetuned] wrote ${OUT_DIR}/success_false_alarm.json"
  echo "[eval_disc_finetuned] wrote ${OUT_DIR}/offline_success_false_alarm.json"
  echo "[eval_disc_finetuned] wrote ${OUT_DIR}/gt_fail_detection.json"
else
  echo "[eval_disc_finetuned] quantitative evaluation disabled (RUN_EVAL=${RUN_EVAL})"
fi

if [[ "${GENERATE_VISUALS_FLAG}" == "true" ]]; then
  mkdir -p "${VIS_OUT_DIR}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-${VIS_OUT_DIR}/.matplotlib}"
  mkdir -p "${MPLCONFIGDIR}"

  echo "[eval_disc_finetuned] generating visualization bundle"
  echo "[eval_disc_finetuned] task=${TASK} split=${SPLIT} seed=${SEED} num_trajs=${NUM_TRAJS}"
  echo "[eval_disc_finetuned] visualization=${VIS_OUT_DIR} device=${DEVICE}"
  echo "[eval_disc_finetuned] offline_episodes=${OFFLINE_EPISODES} pools=offline,offline-success"

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
    --out-dir "${VIS_OUT_DIR}" \
    --pdf-name "finetuned_scores.pdf" \
    --device "${DEVICE}" \
    --encode-batch-size "${ENCODE_BATCH_SIZE}" \
    --delta "${DELTA}" \
    --camera-name "${CAMERA_NAME}" \
    --feature-source transformer \
    --transformer-layer 1 \
    --seed "${SEED}" \
    --fps "${FPS}" \
    --border-thickness "${BORDER_THICKNESS}" \
    "${EXTRA_ARGS[@]}"

  echo "[eval_disc_finetuned] visualization bundle=${VIS_OUT_DIR}"
  echo "[eval_disc_finetuned] wrote ${VIS_OUT_DIR}/finetuned_scores.pdf"
  echo "[eval_disc_finetuned] wrote ${VIS_OUT_DIR}/finetuned_scores_offline.pdf"
  echo "[eval_disc_finetuned] wrote ${VIS_OUT_DIR}/finetuned_scores_offline-success.pdf"
  echo "[eval_disc_finetuned] wrote sampled MP4 files under ${VIS_OUT_DIR}/videos"
else
  echo "[eval_disc_finetuned] visualization bundle disabled (GENERATE_VISUALS=${GENERATE_VISUALS})"
fi
