#!/usr/bin/env bash
# Warm-start the nnPU head from a calibrated checkpoint using extracted
# pretrain latents and policy-only sections from offline collection episodes.

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
USE_ONLY_OFFLINE="${USE_ONLY_OFFLINE:-false}"

EPOCHS=50
LR="${LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
SEED="${SEED:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
RUN_ROOT="${RUN_ROOT:-./outputs/discriminator-finetune}"
RUN_SUBFIX="${RUN_SUBFIX:-}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-tensorboard}"


if [[ -z "${NNPU_CKPT}" ]]; then
  NNPU_CKPT=""
fi
if [[ "${NNPU_CKPT}" != "" && "${NNPU_CKPT}" != "null" && ! -f "${NNPU_CKPT}" ]]; then
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
  "seed=${SEED}"
  "env.environment=${TASK}"
  "algorithm.discriminator.encoder_ckpt=${NNPU_ENCODER_CKPT}"
  "algorithm.discriminator.learner_device=${DEVICE}"
  "offline.discriminator_finetune.episodes_path=${OFFLINE_EPISODES}"
  "offline.discriminator_finetune.pretrain_dir=${PRETRAIN_DIR}"
  "offline.discriminator_finetune.epochs=${EPOCHS}"
  "offline.discriminator_finetune.lr=${LR}"
  "offline.discriminator_finetune.weight_decay=${WEIGHT_DECAY}"
  "offline.discriminator_finetune.batch_size=${BATCH_SIZE}"
  "offline.discriminator_finetune.encode_batch_size=${ENCODE_BATCH_SIZE}"
  "offline.discriminator_finetune.log_interval=${LOG_INTERVAL}"
  "offline.discriminator_finetune.use_only_offline=${USE_ONLY_OFFLINE}"
  "offline.discriminator_finetune.run_root=${RUN_ROOT}"
  "offline.discriminator_finetune.run_subfix=${RUN_SUBFIX}"
  "logging.tensorboard_dir=${TENSORBOARD_DIR}"
  "logging.use_tensorboard=true"
  "logging.use_wandb=false"
  "${HYDRA_DISABLE_LOG_OVERRIDES[@]}"
)
if [[ "${NNPU_CKPT}" != "" && "${NNPU_CKPT}" != "null" ]]; then
  HYDRA_OVERRIDES+=("algorithm.discriminator.checkpoint=${NNPU_CKPT}")
fi
if [[ -n "${NNPU_CAMERA_TO_VIEW}" ]]; then
  HYDRA_OVERRIDES+=("algorithm.discriminator.camera_to_view=${NNPU_CAMERA_TO_VIEW}")
fi

echo "[robosuite][pu_bce] task=${TASK} device=${DEVICE} seed=${SEED}"
if [[ "${NNPU_CKPT}" != "" && "${NNPU_CKPT}" != "null" ]]; then
  echo "[robosuite][pu_bce] parent_ckpt=${NNPU_CKPT} mode=from_checkpoint"
else
  echo "[robosuite][pu_bce] mode=from_init"
fi
echo "[robosuite][pu_bce] episodes=${OFFLINE_EPISODES} pretrain_dir=${PRETRAIN_DIR}"
echo "[robosuite][pu_bce] use_only_offline=${USE_ONLY_OFFLINE}"
echo "[robosuite][pu_bce] epochs=${EPOCHS} lr=${LR} weight_decay=${WEIGHT_DECAY} batch_size=${BATCH_SIZE}"
echo "[robosuite][pu_bce] run_root=${RUN_ROOT} run_subfix=${RUN_SUBFIX:-<none>}"

"${PY}" -m robosuite.pipeline.offline.src.finetune_disc \
  "${HYDRA_OVERRIDES[@]}" \
  "$@"
