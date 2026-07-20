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
NNPU_CKPT="${NNPU_CKPT:-checkpoints/dyn_disc/pu_bce_eval_robosuite-chunk_v2-P98/run_20260719_204315_PickPlaceCereal/checkpoints/pu_bce_head.pth}"
NNPU_ENCODER_CKPT="${NNPU_ENCODER_CKPT:-checkpoints/dyn_disc/dynamics/dinov3_dyn_robosuite-20260619_024518/checkpoint/model_10.pth}"
NNPU_CAMERA_TO_VIEW="${NNPU_CAMERA_TO_VIEW:-}"
# -----------------------------------------------------------------------

OFFLINE_EPISODES="${OFFLINE_EPISODES:-data/${TASK}/offline_data/offline_episodes.pt}"
PRETRAIN_DIR="${PRETRAIN_DIR:-data/${TASK}/discriminator-pretrain-quadratic-c2-l1e2-v2}"

RUN_ROOT="${RUN_ROOT:-./outputs/dipole-rl-offline_disc}"
RUN_SUBFIX="${RUN_SUBFIX:-}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-tensorboard}"
PIPELINE_RUN_DIR="${PIPELINE_RUN_DIR:-}"


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
if [[ -n "${RUN_SUBFIX}" && ! "${RUN_SUBFIX}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "[ERROR] RUN_SUBFIX may contain only letters, digits, underscore, dot, and dash." >&2
  exit 1
fi

if [[ -n "${PIPELINE_RUN_DIR}" ]]; then
  PIPELINE_RUN_DIR="$(realpath -m "${PIPELINE_RUN_DIR}")"
else
  mkdir -p "${RUN_ROOT}"
  RUN_ROOT="$(realpath "${RUN_ROOT}")"
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  PIPELINE_NAME="${TASK}_${TIMESTAMP}"
  if [[ -n "${RUN_SUBFIX}" ]]; then
    PIPELINE_NAME="${PIPELINE_NAME}_${RUN_SUBFIX}"
  fi
  PIPELINE_RUN_DIR="${RUN_ROOT}/${PIPELINE_NAME}"
fi
STAGE_RUN_DIR="${PIPELINE_RUN_DIR}/discriminator"
if [[ -e "${PIPELINE_RUN_DIR}" ]]; then
  echo "[ERROR] Pipeline run directory already exists: ${PIPELINE_RUN_DIR}" >&2
  exit 1
fi
mkdir -p "${PIPELINE_RUN_DIR}"

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
  "offline.discriminator_finetune.run_dir=${STAGE_RUN_DIR}"
  "logging.tensorboard_dir=${TENSORBOARD_DIR}"
  "logging.use_tensorboard=true"
  "logging.use_wandb=false"
  "${HYDRA_DISABLE_LOG_OVERRIDES[@]}"
)
HYDRA_OVERRIDES+=("algorithm.discriminator.checkpoint=${NNPU_CKPT}")

# Keep discriminator.yaml as the source of truth for training defaults. These
# environment variables only take effect when the caller explicitly sets them.
append_optional_override() {
  local env_name="$1"
  local config_key="$2"
  if [[ -v "${env_name}" ]]; then
    HYDRA_OVERRIDES+=("${config_key}=${!env_name}")
  fi
}
append_optional_override EPOCHS offline.discriminator_finetune.epochs
append_optional_override SCHEDULER_HORIZON_EPOCHS offline.discriminator_finetune.scheduler_horizon_epochs
append_optional_override LR offline.discriminator_finetune.lr
append_optional_override WEIGHT_DECAY offline.discriminator_finetune.weight_decay
append_optional_override ENCODE_BATCH_SIZE offline.discriminator_finetune.encode_batch_size
append_optional_override LOG_INTERVAL offline.discriminator_finetune.log_interval
append_optional_override SEED seed
append_optional_override G_NORMALIZATION_ENABLED offline.discriminator_finetune.objective.logit_normalization.enabled
append_optional_override LAMBDA_PRE offline.discriminator_finetune.objective.terms.nnpu_replay.weight
append_optional_override LAMBDA_P offline.discriminator_finetune.objective.terms.gt_positive.weight
append_optional_override LAMBDA_N offline.discriminator_finetune.objective.terms.gt_negative.weight
append_optional_override GT_POSITIVE_TYPE offline.discriminator_finetune.objective.terms.gt_positive.type
append_optional_override GT_POSITIVE_BATCH_SIZE offline.discriminator_finetune.objective.terms.gt_positive.batch_size
append_optional_override GT_NEGATIVE_BATCH_SIZE offline.discriminator_finetune.objective.terms.gt_negative.batch_size
append_optional_override QUADRATIC_CAP_ENABLED offline.discriminator_finetune.objective.quadratic_logit_cap.enabled
append_optional_override QUADRATIC_CAP_C offline.discriminator_finetune.objective.quadratic_logit_cap.cap
append_optional_override QUADRATIC_CAP_LAMBDA offline.discriminator_finetune.objective.quadratic_logit_cap.weight

if [[ "${GT_POSITIVE_TYPE:-positive_logistic}" == "positive_safety_margin" ]]; then
  HYDRA_OVERRIDES+=(
    "+offline.discriminator_finetune.objective.terms.gt_positive.safety_margin_weight=${SAFETY_MARGIN_WEIGHT:-1.0}"
    "+offline.discriminator_finetune.objective.terms.gt_positive.margin_delta=${SAFETY_MARGIN_DELTA:-1.0}"
    "+offline.discriminator_finetune.objective.terms.gt_positive.temperature=${SAFETY_MARGIN_TEMPERATURE:-1.0}"
    "+offline.discriminator_finetune.objective.terms.gt_positive.boundary_source=${SAFETY_MARGIN_BOUNDARY_SOURCE:-parent_checkpoint}"
  )
elif [[ -v SAFETY_MARGIN_WEIGHT || -v SAFETY_MARGIN_DELTA || -v SAFETY_MARGIN_TEMPERATURE || -v SAFETY_MARGIN_BOUNDARY_SOURCE ]]; then
  echo "[ERROR] Safety-margin overrides require GT_POSITIVE_TYPE=positive_safety_margin." >&2
  exit 1
fi

if [[ -n "${NNPU_CAMERA_TO_VIEW}" ]]; then
  HYDRA_OVERRIDES+=("algorithm.discriminator.camera_to_view=${NNPU_CAMERA_TO_VIEW}")
fi

echo "[robosuite][pu_bce] task=${TASK} device=${DEVICE} seed=${SEED:-<yaml>}"
echo "[robosuite][pu_bce] parent_ckpt=${NNPU_CKPT} mode=warm_start"
echo "[robosuite][pu_bce] episodes=${OFFLINE_EPISODES} pretrain_dir=${PRETRAIN_DIR}"
echo "[robosuite][pu_bce] training defaults=discriminator.yaml (explicit environment overrides are preserved)"
echo "[robosuite][pu_bce] pipeline_run_dir=${PIPELINE_RUN_DIR}"
echo "[robosuite][pu_bce] stage_run_dir=${STAGE_RUN_DIR}"

"${PY}" -m robosuite.pipeline.offline.src.finetune_disc \
  "${HYDRA_OVERRIDES[@]}" \
  "$@"
