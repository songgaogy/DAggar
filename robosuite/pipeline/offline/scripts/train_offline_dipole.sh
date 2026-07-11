#!/usr/bin/env bash
# Offline DIPOLE training entry (PROMPT.md [tbd] Offline DIPOLE). Sequential:
#   Phase A  finetune the pretrained IQL critics (unfrozen) on collected policy
#            sections mixed with the warmup transitions; save iql_state_finetuned.pt.
#   Phase B  freeze IQL, precompute TD advantage, run routed weighted-BC on the two
#            flow policies (policy sections advantage-weighted, human -> pos branch,
#            policy_action-during-intervention -> neg branch).
# Data source: data/<task>/offline_data/offline_episodes.pt (collect_data.sh).
# Result dir : data/dipole-rl-offline/<task>_<timestamp>_<postfix>.
#
# Experiment constants below are intentionally hardcoded; edit for another task,
# base policy, nnPU artifact, or IQL warmup state.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=0

# -------------------------------------
TASK="PickPlaceCereal"
POLICY_CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth"
IQL_CKPT="outputs/dipole_rl-iql/offline_iql_qv-v2/PickPlaceCereal/iql_state.pt"   # Pretrained IQL to CONTINUE training from (init_iql_qv.sh output).
OFFLINE_EPISODES="data/${TASK}/offline_data/offline_episodes.pt"  # Collected offline episodes (collect_data.sh output).
PRETRAIN_DATA="data/${TASK}/pretrain_data-20260615_174814"
DEVICE="${DEVICE:-cuda:0}"
NUM_TRAIN_STEPS=15000
IQL_FINETUNE_STEPS=20000
BATCH_SIZE=256
BRANCH_BETA=4
RUN_SUBFIX="beta${BRANCH_BETA}"

# SKIP_RL=1: skip Phase A IQL finetune; reuse IQL_FINETUNED and copy it into the
# new run dir as checkpoints/iql_state_finetuned.pt (see train_offline_dipole.py).
SKIP_RL=1
IQL_FINETUNED="outputs/dipole-rl-offline/PickPlaceCereal_20260710_180410_beta4/checkpoints/iql_state_finetuned.pt"
# -------------------------------------

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
  "algorithm.trainer.batch_size=${BATCH_SIZE}"
  "offline.episodes_path=${OFFLINE_EPISODES}"
  "offline.pretrain_data_path=${PRETRAIN_DATA}"
  "offline.num_train_steps=${NUM_TRAIN_STEPS}"
  "offline.iql_finetune.num_steps=${IQL_FINETUNE_STEPS}"
  "offline.iql_finetune.batch_size=${BATCH_SIZE}"
  "offline.branch_weight.beta=${BRANCH_BETA}"
  "offline.run_subfix=${RUN_SUBFIX}"
  "offline.skip_rl=${SKIP_RL}"
  "offline.iql_finetuned_path=${IQL_FINETUNED}"
)
HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

echo "[train_offline_dipole] task=${TASK} device=${DEVICE} policy_steps=${NUM_TRAIN_STEPS} iql_steps=${IQL_FINETUNE_STEPS} batch_size=${BATCH_SIZE}"
echo "[train_offline_dipole] run_subfix=${RUN_SUBFIX} branch_beta=${BRANCH_BETA}"
echo "[train_offline_dipole] policy_ckpt=${POLICY_CKPT}"
echo "[train_offline_dipole] nnpu_ckpt=${NNPU_CKPT}"
echo "[train_offline_dipole] iql_ckpt=${IQL_CKPT}"
echo "[train_offline_dipole] skip_rl=${SKIP_RL} iql_finetuned=${IQL_FINETUNED}"
echo "[train_offline_dipole] episodes=${OFFLINE_EPISODES}"
echo "[train_offline_dipole] pretrain_data=${PRETRAIN_DATA}"

"${PY}" -m robosuite.pipeline.offline.src.train_offline_dipole "${HYDRA_OVERRIDES[@]}"
