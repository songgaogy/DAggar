#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DINO_V2_CKPT="${DINO_V2_CKPT:-}"
if [[ -z "${DINO_V2_CKPT}" ]]; then
  echo "Please set DINO_V2_CKPT to your pretrained DINOv2 weight path."
  exit 1
fi

python -m robosuite.discriminator.float_dino.eval_discriminator \
  data.expert_dir="${ROOT_DIR}/data/PandaPickPlaceCan/expert_recover" \
  data.fail_rollout_dir="${ROOT_DIR}/data/PandaPickPlaceCan/fail_rollout" \
  data.camera_name="agentview" \
  policy.pretrained_path="${DINO_V2_CKPT}" \
  policy.model_name="vit_base" \
  policy.device="cuda" \
  policy.image_size=518 \
  policy.latent_batch_size=32 \
  output.save_json_path="${ROOT_DIR}/outputs/float_dino_eval_pickplacecan/summary.json" \
  "$@"
