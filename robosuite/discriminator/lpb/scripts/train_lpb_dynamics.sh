#!/usr/bin/env bash
# Train a single LPB dynamics model on pooled expert data from all 6 tasks.
# Run from repo root:
#   bash robosuite/discriminator/lpb/scripts/train_lpb_dynamics.sh
# Override via env vars or pass Hydra-style args, e.g.:
#   EPOCHS=20 BATCH_SIZE=128 \
#     bash robosuite/discriminator/lpb/scripts/train_lpb_dynamics.sh \
#         training.save_freq=5
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${ROOT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Space-separated list of tasks whose expert/ dirs get pooled together.
TASKS="${TASKS:-PandaLift PandaPickPlaceCan PandaStack PickPlaceBread PickPlaceCereal PickPlaceMilk}"

# Build a Hydra list literal: [path1,path2,...]
EXPERT_PATHS=""
for task in ${TASKS}; do
    path="${ROOT_DIR}/data/${task}/expert"
    if [[ -z "${EXPERT_PATHS}" ]]; then
        EXPERT_PATHS="${path}"
    else
        EXPERT_PATHS="${EXPERT_PATHS},${path}"
    fi
done

CAMERA_NAME="${CAMERA_NAME:-agentview}"
HORIZON="${HORIZON:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SAVE_FREQ="${SAVE_FREQ:-10}"
EXPERT_RATIO="${EXPERT_RATIO:-0.5}"

SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints/lpb/dynamics}"
SAVE_NAME="${SAVE_NAME:-dynamics_model.pt}"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m robosuite.discriminator.lpb.train_dynamics \
    "data.expert_paths=[${EXPERT_PATHS}]" \
    "data.rollout_paths=[]" \
    data.camera_name="${CAMERA_NAME}" \
    data.horizon="${HORIZON}" \
    data.image_size="${IMAGE_SIZE}" \
    encoder.ckpt_path="${ROOT_DIR}/robosuite/RL/models/resnet18-f37072fd.pth" \
    encoder.trainable=false \
    training.device="cuda" \
    training.batch_size="${BATCH_SIZE}" \
    training.epochs="${EPOCHS}" \
    training.save_freq="${SAVE_FREQ}" \
    training.expert_ratio="${EXPERT_RATIO}" \
    save_dir="${SAVE_DIR}" \
    save_name="${SAVE_NAME}" \
    "$@"
