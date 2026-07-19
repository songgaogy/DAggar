#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# -------------------------------------
TASK="PickPlaceCereal"
POLICY_CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"
PIPELINE_RUN_DIR="${PIPELINE_RUN_DIR:-outputs/dipole-rl-offline_disc/PickPlaceCereal_20260719_212234}"
BRANCH_BETA=1
USE_ONLINE_SUCCESS=0
NUM_TRAIN_STEPS=15000
# -------------------------------------

OFFLINE_EPISODES="${OFFLINE_EPISODES:-data/${TASK}/offline_data/offline_episodes.pt}"
PRETRAIN_DATA="${PRETRAIN_DATA:-data/${TASK}/pretrain_data-20260615_174814}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-256}"
PREENCODE_BATCH_SIZE="${PREENCODE_BATCH_SIZE:-64}"

if [[ ! -d "${PIPELINE_RUN_DIR}" ]]; then
  echo "[ERROR] PIPELINE_RUN_DIR does not exist: ${PIPELINE_RUN_DIR}" >&2
  exit 1
fi
PIPELINE_RUN_DIR="$(realpath "${PIPELINE_RUN_DIR}")"
DISCRIMINATOR_RUN_DIR="${PIPELINE_RUN_DIR}/discriminator"
NNPU_CKPT="${DISCRIMINATOR_RUN_DIR}/checkpoints/pu_bce_head_finetuned.pth"
STAGE_RUN_DIR="${PIPELINE_RUN_DIR}"
CHECKPOINT_DIR="${STAGE_RUN_DIR}/checkpoints"
if [[ ! -d "${DISCRIMINATOR_RUN_DIR}" ]]; then
  echo "[ERROR] Discriminator stage directory does not exist: ${DISCRIMINATOR_RUN_DIR}" >&2
  exit 1
fi
if [[ ! -f "${NNPU_CKPT}" ]]; then
  echo "[ERROR] Finetuned discriminator checkpoint does not exist: ${NNPU_CKPT}" >&2
  exit 1
fi
if [[ -e "${CHECKPOINT_DIR}" ]]; then
  echo "[ERROR] DIPOLE checkpoints directory already exists: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1
# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

HYDRA_OVERRIDES=(
  "seed=${SEED}"
  "env.environment=${TASK}"
  "runtime.init_checkpoint=${POLICY_CKPT}"
  "algorithm.discriminator.checkpoint=${NNPU_CKPT}"
  "algorithm.discriminator.inference.device=${DEVICE}"
  "algorithm.discriminator.learner_device=${DEVICE}"
  "algorithm.flow.device=${DEVICE}"
  "algorithm.flow.inference_device=${DEVICE}"
  "algorithm.trainer.batch_size=${POLICY_BATCH_SIZE}"
  "offline.episodes_path=${OFFLINE_EPISODES}"
  "offline.pretrain_data_path=${PRETRAIN_DATA}"
  "offline.num_train_steps=${NUM_TRAIN_STEPS}"
  "offline.preencode_batch_size=${PREENCODE_BATCH_SIZE}"
  "offline.branch_weight.beta=${BRANCH_BETA}"
  "offline.use_online_success=${USE_ONLINE_SUCCESS}"
  "offline.run_dir=${STAGE_RUN_DIR}"
  "logging.tensorboard_dir=tensorboard/dipole"
  "${HYDRA_DISABLE_LOG_OVERRIDES[@]}"
)

echo "[train_offline_dipole] task=${TASK} device=${DEVICE} seed=${SEED} policy_steps=${NUM_TRAIN_STEPS}"
echo "[train_offline_dipole] algorithm=discriminator_weighted_offline_dipole branch_beta=${BRANCH_BETA}"
echo "[train_offline_dipole] pipeline_run_dir=${PIPELINE_RUN_DIR}"
echo "[train_offline_dipole] discriminator_run_dir=${DISCRIMINATOR_RUN_DIR}"
echo "[train_offline_dipole] stage_run_dir=${STAGE_RUN_DIR}"
echo "[train_offline_dipole] policy_batch=${POLICY_BATCH_SIZE} preencode_batch=${PREENCODE_BATCH_SIZE}"
echo "[train_offline_dipole] policy_ckpt=${POLICY_CKPT}"
echo "[train_offline_dipole] discriminator_ckpt=${NNPU_CKPT}"
echo "[train_offline_dipole] episodes=${OFFLINE_EPISODES}"
echo "[train_offline_dipole] pretrain_data=${PRETRAIN_DATA}"
echo "[train_offline_dipole] use_online_success=${USE_ONLINE_SUCCESS}"

"${PY}" -m robosuite.pipeline.offline.src.train_offline_dipole "${HYDRA_OVERRIDES[@]}"
