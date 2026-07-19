#!/usr/bin/env bash
# Offline DAgger positive-only training entry.
# Uses human intervention sections from offline_episodes.pt plus expert pretrain
# demos. Only the positive DIPOLE policy is updated; the negative policy is kept
# unchanged in the output checkpoint for eval_offline_dipole.sh compatibility.
#
# USE_ONLINE_SUCCESS=1 additionally folds in entire pure on-policy success
# rollouts (terminal_reason==success AND zero human intervention) as positive BC
# demos, on top of human + pretrain. USE_ONLINE_SUCCESS=0 -> original behavior.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=1

# -------------------------------------
TASK="PickPlaceCereal"
POLICY_CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"
OFFLINE_EPISODES="data/${TASK}/offline_data/offline_episodes.pt"
PRETRAIN_DATA="data/${TASK}/pretrain_data-20260615_174814"
DEVICE="cuda:0"
NUM_TRAIN_STEPS=15000
BATCH_SIZE=256
MAX_PRETRAIN_TRAJECTORIES=null
RUN_SUBFIX="dagger-with_success"
USE_ONLINE_SUCCESS=1        # add success online rollout with no human intervention
# -------------------------------------

RUN_ROOT="outputs/dipole-rl-offline_disc"
LOGGING_USE_TENSORBOARD="${LOGGING_USE_TENSORBOARD:-true}"
LOGGING_USE_WANDB="${LOGGING_USE_WANDB:-false}"

# Normalize USE_ONLINE_SUCCESS (1/0/true/false) into a Hydra bool.
case "${USE_ONLINE_SUCCESS:-0}" in
  1|true|True|TRUE) USE_ONLINE_SUCCESS_BOOL=true ;;
  *)                USE_ONLINE_SUCCESS_BOOL=false ;;
esac

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1

# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

HYDRA_OVERRIDES=(
  "env.environment=${TASK}"
  "runtime.init_checkpoint=${POLICY_CKPT}"
  "algorithm.flow.device=${DEVICE}"
  "algorithm.flow.inference_device=${DEVICE}"
  "algorithm.trainer.batch_size=${BATCH_SIZE}"
  "offline.episodes_path=${OFFLINE_EPISODES}"
  "offline.pretrain_data_path=${PRETRAIN_DATA}"
  "offline.max_pretrain_trajectories=${MAX_PRETRAIN_TRAJECTORIES}"
  "offline.use_online_success=${USE_ONLINE_SUCCESS_BOOL}"
  "offline.num_train_steps=${NUM_TRAIN_STEPS}"
  "offline.run_root=${RUN_ROOT}"
  "offline.run_subfix=${RUN_SUBFIX}"
  "logging.use_tensorboard=${LOGGING_USE_TENSORBOARD}"
  "logging.use_wandb=${LOGGING_USE_WANDB}"
)
HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

echo "[train_dagger] task=${TASK} device=${DEVICE} steps=${NUM_TRAIN_STEPS} batch_size=${BATCH_SIZE}"
echo "[train_dagger] run_root=${RUN_ROOT} run_subfix=${RUN_SUBFIX}"
echo "[train_dagger] policy_ckpt=${POLICY_CKPT}"
echo "[train_dagger] episodes=${OFFLINE_EPISODES}"
echo "[train_dagger] pretrain_data=${PRETRAIN_DATA} max_pretrain_trajectories=${MAX_PRETRAIN_TRAJECTORIES}"
echo "[train_dagger] use_online_success=${USE_ONLINE_SUCCESS_BOOL} (from USE_ONLINE_SUCCESS=${USE_ONLINE_SUCCESS:-0})"
echo "[train_dagger] logging tensorboard=${LOGGING_USE_TENSORBOARD} wandb=${LOGGING_USE_WANDB}"

"${PY}" -m robosuite.pipeline.offline.src.train_dagger "${HYDRA_OVERRIDES[@]}" "$@"
