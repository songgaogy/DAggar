#!/usr/bin/env bash
# Success-only SFT for the DIPOLE flow policy (positive-branch only). Fine-tunes
# the pretrained flow policy on the same positive data as naive / neg_all
# (expert pretrain + success_rollout pre-success frames), training just the
# positive LoRA branch — no fail data and no IQL / nnPU / discriminator /
# advantage. Config: config/success_only.yaml.
#
# Experiment constants below are intentionally hardcoded; edit for another task
# or base policy.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=0

# -------------------------------------
TASK="PickPlaceCereal"
POLICY_CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"   # frozen base flow policy
DEVICE="${DEVICE:-cuda:0}"
NUM_TRAIN_STEPS=15000
RUN_SUBFIX="success_only"
# -------------------------------------

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1
# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

HYDRA_OVERRIDES=(
  "env.environment=${TASK}"
  "runtime.init_checkpoint=${POLICY_CKPT}"
  "algorithm.flow.device=${DEVICE}"
  "algorithm.flow.inference_device=${DEVICE}"
  "offline.num_train_steps=${NUM_TRAIN_STEPS}"
  "offline.run_subfix=${RUN_SUBFIX}"
)
HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

echo "[train_success_only] task=${TASK} device=${DEVICE} steps=${NUM_TRAIN_STEPS}"
echo "[train_success_only] run_subfix=${RUN_SUBFIX}"
echo "[train_success_only] policy_ckpt=${POLICY_CKPT}"

"${PY}" -m robosuite.pipeline.offline.legacy.train_success_only "${HYDRA_OVERRIDES[@]}"
