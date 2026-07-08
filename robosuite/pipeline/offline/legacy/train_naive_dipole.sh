#!/usr/bin/env bash
# Naive offline DIPOLE training (offline.mode=naive). A hard-label baseline that
# skips the whole IQL / nnPU / advantage stack: expert (pretrain_dir) +
# success_rollout frames train the positive branch (w_pos=1), fail_rollout frames
# train the negative branch (w_neg=1). Same entry as train_offline_dipole.sh,
# only the mode + data selection differ.
#
# Experiment constants below are intentionally hardcoded; edit for another task
# or base policy. No nnPU / IQL checkpoints are needed in naive mode.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=1

# -------------------------------------
TASK="PickPlaceCereal"
POLICY_CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"
DEVICE="${DEVICE:-cuda:0}"
NUM_TRAIN_STEPS=15000
RUN_SUBFIX="naive"
# -------------------------------------

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1

HYDRA_OVERRIDES=(
  "offline.mode=naive"
  "env.environment=${TASK}"
  "runtime.init_checkpoint=${POLICY_CKPT}"
  "algorithm.flow.device=${DEVICE}"
  "algorithm.flow.inference_device=${DEVICE}"
  "offline.num_train_steps=${NUM_TRAIN_STEPS}"
  "offline.run_subfix=${RUN_SUBFIX}"
)

echo "[train_naive_dipole] mode=naive task=${TASK} device=${DEVICE} steps=${NUM_TRAIN_STEPS}"
echo "[train_naive_dipole] run_subfix=${RUN_SUBFIX}"
echo "[train_naive_dipole] policy_ckpt=${POLICY_CKPT}"

"${PY}" -m robosuite.pipeline.offline.train_offline_dipole "${HYDRA_OVERRIDES[@]}"
