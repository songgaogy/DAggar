#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m robosuite.discriminator.lpb.train_dynamics \
  "data.expert_paths=[${ROOT_DIR}/data/PandaLift/expert]" \
  "data.rollout_paths=[${ROOT_DIR}/data/PandaLift/unlabled,${ROOT_DIR}/data/PandaLift/fail_rollout]" \
  data.camera_name="agentview" \
  data.horizon=1 \
  data.image_size=256 \
  encoder.ckpt_path="${ROOT_DIR}/robosuite/RL/models/resnet18-f37072fd.pth" \
  encoder.trainable=false \
  training.device="cuda" \
  training.batch_size=256 \
  training.epochs=20 \
  training.save_freq=5 \
  training.expert_ratio=0.5 \
  save_dir="${ROOT_DIR}/checkpoints/PandaLift/discriminator/lpb_dynamics"
