#!/usr/bin/env bash
# Offline CFGRL: DAgger good-data positive policy plus an independently trained
# negative policy. CFG_MODE=all uses uniform Phase-B data; CFG_MODE=neg-weighted
# reuses Offline DIPOLE's routed V/GAE negative weights.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=0

# -------------------------------------
TASK="PickPlaceCereal"
POLICY_CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth"
IQL_CKPT="outputs/dipole_rl-iql/offline_iql_qv-v2_explore-ensemble/PickPlaceCereal/iql_state.pt"   # Pretrained IQL to CONTINUE training from (init_iql_qv.sh output).
OFFLINE_EPISODES="data/${TASK}/offline_data/offline_episodes.pt"  # Collected offline episodes (collect_data.sh output).
PRETRAIN_DATA="data/${TASK}/pretrain_data-20260615_174814"
DEVICE="${DEVICE:-cuda:0}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-15000}"
V_FINETUNE_STEPS="${V_FINETUNE_STEPS:-10000}"
V_BATCH_SIZE="${V_BATCH_SIZE:-512}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-256}"
CFG_MODE="${CFG_MODE:-all}"      # all or neg-weighted

# w_pos = sigmoid(beta * (G + k)); decision boundary at G=-k.
BRANCH_BETA="${BRANCH_BETA:-1}"
BRANCH_K="${BRANCH_K:--2}"
RUN_SUBFIX="beta${BRANCH_BETA}_k${BRANCH_K}_cfg-${CFG_MODE}"

# SKIP_RL=1: skip Phase A IQL finetune; reuse IQL_FINETUNED and copy it into the
# new run dir as checkpoints/iql_state_finetuned.pt (see train_offline_dipole.py).
SKIP_RL="${SKIP_RL:-1}"
IQL_FINETUNED="outputs/dipole-rl-offline/PickPlaceCereal_20260713_113548_beta1_k-2_ensemble-GAE/checkpoints/iql_state_finetuned.pt"
USE_ONLINE_SUCCESS="${USE_ONLINE_SUCCESS:-1}"
# -------------------------------------

case "${CFG_MODE}" in
  all|neg-weighted) ;;
  *)
    echo "[train_offline_cfgrl] CFG_MODE must be 'all' or 'neg-weighted', got '${CFG_MODE}'." >&2
    exit 2
    ;;
esac

case "${USE_ONLINE_SUCCESS}" in
  1|true|True|TRUE) USE_ONLINE_SUCCESS_BOOL=true ;;
  *)                USE_ONLINE_SUCCESS_BOOL=false ;;
esac

case "${SKIP_RL}" in
  1|true|True|TRUE) SKIP_RL_BOOL=true ;;
  *)                SKIP_RL_BOOL=false ;;
esac

RUN_ROOT="${RUN_ROOT:-outputs/dipole-rl-offline}"
MAX_PRETRAIN_TRAJECTORIES="${MAX_PRETRAIN_TRAJECTORIES:-null}"
LOGGING_USE_TENSORBOARD="${LOGGING_USE_TENSORBOARD:-true}"
LOGGING_USE_WANDB=false

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1

# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

HYDRA_OVERRIDES=(
  "env.environment=${TASK}"
  "runtime.init_checkpoint=${POLICY_CKPT}"
  "algorithm.discriminator.checkpoint=${NNPU_CKPT}"
  "algorithm.q_learning.warmup_ckpt=${IQL_CKPT}"
  "algorithm.q_learning.config.device=${DEVICE}"
  "algorithm.discriminator.inference.device=${DEVICE}"
  "algorithm.discriminator.learner_device=${DEVICE}"
  "algorithm.flow.device=${DEVICE}"
  "algorithm.flow.inference_device=${DEVICE}"
  "algorithm.trainer.batch_size=${POLICY_BATCH_SIZE}"
  "offline.cfg_mode=${CFG_MODE}"
  "offline.episodes_path=${OFFLINE_EPISODES}"
  "offline.pretrain_data_path=${PRETRAIN_DATA}"
  "offline.max_pretrain_trajectories=${MAX_PRETRAIN_TRAJECTORIES}"
  "offline.num_train_steps=${NUM_TRAIN_STEPS}"
  "offline.iql_finetune.num_steps=${V_FINETUNE_STEPS}"
  "offline.iql_finetune.batch_size=${V_BATCH_SIZE}"
  "offline.branch_weight.beta=${BRANCH_BETA}"
  "offline.branch_weight.k=${BRANCH_K}"
  "offline.use_online_success=${USE_ONLINE_SUCCESS_BOOL}"
  "offline.run_root=${RUN_ROOT}"
  "offline.run_subfix=${RUN_SUBFIX}"
  "offline.skip_rl=${SKIP_RL_BOOL}"
  "logging.use_tensorboard=${LOGGING_USE_TENSORBOARD}"
  "logging.use_wandb=${LOGGING_USE_WANDB}"
)
if [[ "${CFG_MODE}" == "neg-weighted" && "${SKIP_RL_BOOL}" == "true" ]]; then
  HYDRA_OVERRIDES+=("offline.iql_finetuned_path=${IQL_FINETUNED}")
fi
HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

echo "[train_offline_cfgrl] task=${TASK} mode=${CFG_MODE} device=${DEVICE} steps=${NUM_TRAIN_STEPS}"
echo "[train_offline_cfgrl] pos_batch=${POLICY_BATCH_SIZE} neg_batch=${POLICY_BATCH_SIZE}"
echo "[train_offline_cfgrl] policy_ckpt=${POLICY_CKPT}"
echo "[train_offline_cfgrl] episodes=${OFFLINE_EPISODES}"
echo "[train_offline_cfgrl] pretrain_data=${PRETRAIN_DATA} max_pretrain=${MAX_PRETRAIN_TRAJECTORIES}"
echo "[train_offline_cfgrl] use_online_success=${USE_ONLINE_SUCCESS_BOOL}"
if [[ "${CFG_MODE}" == "neg-weighted" ]]; then
  echo "[train_offline_cfgrl] nnpu_ckpt=${NNPU_CKPT}"
  echo "[train_offline_cfgrl] iql_ckpt=${IQL_CKPT} skip_rl=${SKIP_RL_BOOL}"
  echo "[train_offline_cfgrl] branch_beta=${BRANCH_BETA} branch_k=${BRANCH_K}"
fi

"${PY}" -m robosuite.pipeline.offline.src.train_offline_cfgrl \
  "${HYDRA_OVERRIDES[@]}" "$@"
