#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ADAPTIVE_DELTA="${ADAPTIVE_DELTA:-true}"

python -m robosuite.discriminator.lpb.eval_knn_discriminator \
  data.expert_dir="${ROOT_DIR}/data/PandaLift/expert" \
  data.fail_rollout_dir="${ROOT_DIR}/data/PandaLift/fail_rollout" \
  data.camera_name="agentview" \
  model.lpb_ckpt="${ROOT_DIR}/checkpoints/PandaLift/discriminator/lpb_dynamics/dynamics_model_ep0015.pt" \
  feature.device="cuda" \
  detector.device="cuda" \
  detector.delta=10.0 \
  detector.delta_step=1.0 \
  detector.lambda_mode="mean" \
  labels.fail_tail_ratio=0.2 \
  detector.lambda_window_size=10 \
  online.adaptive_delta="${ADAPTIVE_DELTA}" \
  save_dir="${ROOT_DIR}/checkpoints/PandaLift/discriminator/lpb_knn" \
  "$@"
