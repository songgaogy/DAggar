#!/usr/bin/env bash
# Offline DIPOLE training entry (README step-4). Fine-tunes the pretrained flow
# policy on disk data (pretrain expert + filtered offline_data) using the
# DIPOLE-RL branch weighting with a frozen IQL critic and TD advantage — no env
# rollout. Config inherits train_dipole_rl via config/offline.yaml.
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
IQL_CKPT="outputs/dipole_rl/offline_iql_qv-v2/PickPlaceCereal/iql_state.pt"
DEVICE="${DEVICE:-cuda:0}"
NUM_TRAIN_STEPS=1
G_NORMALIZATION=false
RUN_SUBFIX="no-norm"
# -------------------------------------

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1

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
  "offline.num_train_steps=${NUM_TRAIN_STEPS}"
  "algorithm.dipole.use_norm=${G_NORMALIZATION}"
  "offline.run_subfix=${RUN_SUBFIX}"
)

echo "[train_offline_dipole] task=${TASK} device=${DEVICE} steps=${NUM_TRAIN_STEPS}"
echo "[train_offline_dipole] g_normalization(use_norm)=${G_NORMALIZATION} run_subfix=${RUN_SUBFIX}"
echo "[train_offline_dipole] policy_ckpt=${POLICY_CKPT}"
echo "[train_offline_dipole] nnpu_ckpt=${NNPU_CKPT}"
echo "[train_offline_dipole] iql_ckpt=${IQL_CKPT}"

"${PY}" -m robosuite.pipeline.offline.train_offline_dipole "${HYDRA_OVERRIDES[@]}"
