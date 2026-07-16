#!/usr/bin/env bash
# Warm-start the nnPU head from a calibrated checkpoint using extracted
# pretrain latents, policy success segments, and intervention-onset GT windows.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1

# -----------------------------------------------------------------------
TASK="${TASK:-PickPlaceCereal}"
NNPU_CKPT="${NNPU_CKPT:-}"
NNPU_ENCODER_CKPT="${NNPU_ENCODER_CKPT:-checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_10.pth}"
NNPU_CAMERA_TO_VIEW="${NNPU_CAMERA_TO_VIEW:-}"
# -----------------------------------------------------------------------

OFFLINE_EPISODES="${OFFLINE_EPISODES:-data/${TASK}/offline_data/offline_episodes.pt}"
PRETRAIN_DIR="${PRETRAIN_DIR:-data/${TASK}/discriminator-pretrain}"

RUN_ROOT="${RUN_ROOT:-./outputs/discriminator-finetune}"
RUN_SUBFIX="${RUN_SUBFIX:-}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-tensorboard}"


if [[ -z "${NNPU_CKPT}" || "${NNPU_CKPT}" == "null" ]]; then
  echo "[ERROR] NNPU_CKPT is required for warm-start Step 3 finetuning." >&2
  exit 1
fi
if [[ ! -f "${NNPU_CKPT}" ]]; then
  echo "[ERROR] NNPU_CKPT does not exist: ${NNPU_CKPT}" >&2
  exit 1
fi
if [[ "${NNPU_ENCODER_CKPT}" != "null" && -n "${NNPU_ENCODER_CKPT}" && ! -f "${NNPU_ENCODER_CKPT}" ]]; then
  echo "[ERROR] NNPU_ENCODER_CKPT does not exist: ${NNPU_ENCODER_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${OFFLINE_EPISODES}" ]]; then
  echo "[ERROR] OFFLINE_EPISODES does not exist: ${OFFLINE_EPISODES}" >&2
  exit 1
fi
if [[ ! -f "${PRETRAIN_DIR}/manifest.json" ]]; then
  echo "[ERROR] PRETRAIN_DIR must contain manifest.json: ${PRETRAIN_DIR}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

HYDRA_OVERRIDES=(
  "env.environment=${TASK}"
  "algorithm.discriminator.encoder_ckpt=${NNPU_ENCODER_CKPT}"
  "algorithm.discriminator.learner_device=${DEVICE}"
  "offline.discriminator_finetune.episodes_path=${OFFLINE_EPISODES}"
  "offline.discriminator_finetune.pretrain_dir=${PRETRAIN_DIR}"
  "offline.discriminator_finetune.run_root=${RUN_ROOT}"
  "offline.discriminator_finetune.run_subfix=${RUN_SUBFIX}"
  "logging.tensorboard_dir=${TENSORBOARD_DIR}"
  "logging.use_tensorboard=true"
  "logging.use_wandb=false"
  "${HYDRA_DISABLE_LOG_OVERRIDES[@]}"
)
HYDRA_OVERRIDES+=("algorithm.discriminator.checkpoint=${NNPU_CKPT}")

# Keep finetune_disc.yaml as the source of truth for training defaults. These
# environment variables only take effect when the caller explicitly sets them.
append_optional_override() {
  local env_name="$1"
  local config_key="$2"
  if [[ -v "${env_name}" ]]; then
    HYDRA_OVERRIDES+=("${config_key}=${!env_name}")
  fi
}
append_optional_override EPOCHS offline.discriminator_finetune.epochs
append_optional_override LR offline.discriminator_finetune.lr
append_optional_override WEIGHT_DECAY offline.discriminator_finetune.weight_decay
append_optional_override ENCODE_BATCH_SIZE offline.discriminator_finetune.encode_batch_size
append_optional_override LOG_INTERVAL offline.discriminator_finetune.log_interval
append_optional_override SEED seed
append_optional_override LAMBDA_PRE offline.discriminator_finetune.objective.terms.nnpu_replay.weight
append_optional_override LAMBDA_P offline.discriminator_finetune.objective.terms.gt_positive.weight
append_optional_override LAMBDA_N offline.discriminator_finetune.objective.terms.gt_negative.weight
append_optional_override GT_POSITIVE_BATCH_SIZE offline.discriminator_finetune.objective.terms.gt_positive.batch_size
append_optional_override GT_NEGATIVE_BATCH_SIZE offline.discriminator_finetune.objective.terms.gt_negative.batch_size
append_optional_override SAFETY_MARGIN_WEIGHT offline.discriminator_finetune.objective.terms.gt_positive.safety_margin_weight
append_optional_override SAFETY_MARGIN_DELTA offline.discriminator_finetune.objective.terms.gt_positive.margin_delta
append_optional_override SAFETY_MARGIN_TEMPERATURE offline.discriminator_finetune.objective.terms.gt_positive.temperature
append_optional_override SAFETY_MARGIN_BOUNDARY_SOURCE offline.discriminator_finetune.objective.terms.gt_positive.boundary_source

if [[ -n "${NNPU_CAMERA_TO_VIEW}" ]]; then
  HYDRA_OVERRIDES+=("algorithm.discriminator.camera_to_view=${NNPU_CAMERA_TO_VIEW}")
fi

echo "[robosuite][pu_bce] task=${TASK} device=${DEVICE} seed=${SEED:-<yaml>}"
echo "[robosuite][pu_bce] parent_ckpt=${NNPU_CKPT} mode=warm_start"
echo "[robosuite][pu_bce] episodes=${OFFLINE_EPISODES} pretrain_dir=${PRETRAIN_DIR}"
echo "[robosuite][pu_bce] training defaults=finetune_disc.yaml (explicit environment overrides are preserved)"
echo "[robosuite][pu_bce] run_root=${RUN_ROOT} run_subfix=${RUN_SUBFIX:-<none>}"

"${PY}" -m robosuite.pipeline.offline.src.finetune_disc \
  "${HYDRA_OVERRIDES[@]}" \
  "$@"
