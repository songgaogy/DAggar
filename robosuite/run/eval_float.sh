#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
ADAPTIVE_DELTA=true

python -m robosuite.discriminator.float.eval_discriminator \
  data.expert_dir="${ROOT_DIR}/data/PandaPickPlaceCan/expert_recover" \
  data.fail_rollout_dir="${ROOT_DIR}/data/PandaPickPlaceCan/fail_rollout" \
  policy.ckpt="${ROOT_DIR}/checkpoints/PickPlaceCan/flow_unet-10/flow_unet_ep0400_20260318_020356.pt" \
  policy.type="flow_unet" \
  policy.device="cuda" \
  policy.image_size=-1 \
  float.ot_device="cuda" \
  float.num_expert_candidates=100 \
  calibration.delta=10.0 \
  calibration.delta_step=1.0 \
  calibration.max_expert_trajectories=100 \
  labels.fail_tail_ratio=0.5 \
  online.adaptive_delta="${ADAPTIVE_DELTA}" \
  output.save_json_path="${ROOT_DIR}/checkpoints/PickPlaceCan/discriminator/float_unet/summary.json" \
  "$@"
