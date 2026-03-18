#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DINO_V2_CKPT="${DINO_V2_CKPT:-}"
if [[ -z "${DINO_V2_CKPT}" ]]; then
  echo "Please set DINO_V2_CKPT to your pretrained DINOv2 weight path."
  exit 1
fi

NUM_VIS="${NUM_VIS:-5}"

python -m robosuite.discriminator.float_dino.utils.visualize \
  --expert-dir "${ROOT_DIR}/data/PandaPickPlaceCan/expert_recover" \
  --fail-dir "${ROOT_DIR}/data/PandaPickPlaceCan/fail_rollout" \
  --camera-name "agentview" \
  --pretrained-path "${DINO_V2_CKPT}" \
  --model-name "vit_base" \
  --policy-device cuda \
  --image-size 518 \
  --latent-batch-size 32 \
  --num-expert-candidates 50 \
  --num-vis "${NUM_VIS}" \
  --output-dir "${ROOT_DIR}/outputs/float_dino_vis" \
  "$@"
