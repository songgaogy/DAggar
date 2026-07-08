#!/usr/bin/env bash
# Headless success-rate + video eval for an offline-finetuned DIPOLE checkpoint
# (README step-4). Uses DIPOLE two-branch guidance with OMEGA. Results land under
# the training run dir (<run_dir>/eval/...) by default.
#
# Set POLICY_CKPT to the offline-finetuned checkpoint (e.g. .../checkpoints/latest.pt).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=0

# -------------------------------------
TASK="PickPlaceCereal"
POLICY_CKPT="/home/dodo/Documents/DAggar/robosuite/outputs/dipole_offline/PickPlaceCereal/2026-07-08_02-35-39_neg_all/checkpoints/latest.pt"
OMEGAS=(0 1 5)
# OMEGAS=(0.5 2 10)
EVAL_EPISODES=50
EVAL_EPISODE_MAX_STEPS=500
SAVE_VIDEO=true
DEVICE="${DEVICE:-cuda:0}"
SEED=10086
# -------------------------------------

# Optional base flow checkpoint for env metadata (defaults to run_info.json).
INIT_CHECKPOINT=""
VIDEO_SIZE=256    # square video frame size; 256 renders 4x fewer pixels than the old 512

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1

for OMEGA in "${OMEGAS[@]}"; do
  EVAL_ARGS=(
    --checkpoint "${POLICY_CKPT}"
    --env-name "${TASK}"
    --task-name "${TASK}"
    --omega "${OMEGA}"
    --episodes "${EVAL_EPISODES}"
    --episode-max-steps "${EVAL_EPISODE_MAX_STEPS}"
    --video-output "${SAVE_VIDEO}"
    --video-height "${VIDEO_SIZE}"
    --video-width "${VIDEO_SIZE}"
    --seed "${SEED}"
    --device "${DEVICE}"
  )
  if [[ -n "${INIT_CHECKPOINT}" ]]; then
    EVAL_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT}")
  fi

  echo "[eval_offline_dipole] task=${TASK} ckpt=${POLICY_CKPT} omega=${OMEGA} episodes=${EVAL_EPISODES} seed=${SEED} video=${SAVE_VIDEO} video_size=${VIDEO_SIZE}"
  "${PY}" -m robosuite.pipeline.offline.eval_offline_dipole "${EVAL_ARGS[@]}"
done
