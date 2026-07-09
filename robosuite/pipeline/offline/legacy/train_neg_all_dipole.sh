#!/usr/bin/env bash
# neg_all offline DIPOLE training (offline.mode=neg_all). Decoupled hard-label
# baseline that skips the IQL / nnPU / advantage stack:
#   - positive branch (w_pos=1): expert (pretrain_dir) + success_rollout
#   - negative branch (w_neg=1): ALL offline_data (success_rollout + fail_rollout)
# success frames therefore drive BOTH branches. Same entry as
# train_offline_dipole.sh; only the mode differs. No nnPU / IQL checkpoints.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=0

# -------------------------------------
TASK="PickPlaceCereal"
POLICY_CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"
DEVICE="${DEVICE:-cuda:0}"
NUM_TRAIN_STEPS=15000
RUN_SUBFIX="neg_all"
# -------------------------------------

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1
# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

HYDRA_OVERRIDES=(
  "offline.mode=neg_all"
  "env.environment=${TASK}"
  "runtime.init_checkpoint=${POLICY_CKPT}"
  "algorithm.flow.device=${DEVICE}"
  "algorithm.flow.inference_device=${DEVICE}"
  "offline.num_train_steps=${NUM_TRAIN_STEPS}"
  "offline.run_subfix=${RUN_SUBFIX}"
)
HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

echo "[train_neg_all_dipole] mode=neg_all task=${TASK} device=${DEVICE} steps=${NUM_TRAIN_STEPS}"
echo "[train_neg_all_dipole] run_subfix=${RUN_SUBFIX}"
echo "[train_neg_all_dipole] policy_ckpt=${POLICY_CKPT}"

"${PY}" -m robosuite.pipeline.offline.train_offline_dipole "${HYDRA_OVERRIDES[@]}"
