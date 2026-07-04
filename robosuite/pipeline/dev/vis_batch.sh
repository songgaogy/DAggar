#!/usr/bin/env bash
# Visualize offline DIPOLE training batches. Rebuilds the exact offline pipeline
# (frozen IQL critic + nnPU + precomputed TD advantage), samples a few training
# batches straight from the mixed replay buffer, and renders each as one short
# MP4: a grid of chunk clips playing in sync with per-sample G / A / failure HUD.
# Sanity tool for checking whether the IQL per-batch weight (G) is sensible.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=0

# ----------------------------------------------------------------------------------
TASK="PickPlaceCereal"
POLICY_CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth"
IQL_CKPT="outputs/dipole_rl/offline_iql_qv-v2/PickPlaceCereal/iql_state.pt"

VIS_BATCH_SIZE=6        # numbers sample from each batch, put them into the same short video
VIS_NUM_BATCHES=8       # number of batches to sample from, each output a video
OUT_DIR="outputs/dipole_offline/${TASK}/dev/vis_batch"  # output directory
DEVICE="${DEVICE:-cuda:0}"
VIS_FPS="${VIS_FPS:-4}"        # chunk plays slow so frames are inspectable
VIS_SEED="${VIS_SEED:-0}"
VIS_CELL_PX="${VIS_CELL_PX:-384}"  # per-sample cell resolution in the grid
# ----------------------------------------------------------------------------------

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
  "+vis.batch_size=${VIS_BATCH_SIZE}"
  "+vis.num_batches=${VIS_NUM_BATCHES}"
  "+vis.out_dir=${OUT_DIR}"
  "+vis.fps=${VIS_FPS}"
  "+vis.seed=${VIS_SEED}"
  "+vis.cell_px=${VIS_CELL_PX}"
)

echo "[vis_batch] task=${TASK} device=${DEVICE}"
echo "[vis_batch] vis_batch_size=${VIS_BATCH_SIZE} num_batches=${VIS_NUM_BATCHES} fps=${VIS_FPS}"
echo "[vis_batch] policy_ckpt=${POLICY_CKPT}"
echo "[vis_batch] nnpu_ckpt=${NNPU_CKPT}"
echo "[vis_batch] iql_ckpt=${IQL_CKPT}"
echo "[vis_batch] out_dir=${OUT_DIR}"

"${PY}" -m robosuite.pipeline.dev.vis_batch "${HYDRA_OVERRIDES[@]}"
