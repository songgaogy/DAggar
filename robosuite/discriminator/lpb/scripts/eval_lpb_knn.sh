#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/dodo/Documents/DAggar/robosuite"
PYTHON_BIN="/home/dodo/miniconda3/envs/daggar/bin/python"

export CUDA_VISIBLE_DEVICES=0

LPB_CKPT_PATH="${LPB_CKPT_PATH:-${ROOT_DIR}/checkpoints/PandaLift/discriminator/lpb_dynamics/contrastive}"
ADAPTIVE_DELTA=true

cd "${ROOT_DIR}"

exec "${PYTHON_BIN}" -m robosuite.discriminator.lpb.eval_knn_discriminator \
  data.expert_dir="${ROOT_DIR}/data/PandaLift/expert" \
  data.fail_rollout_dir="${ROOT_DIR}/data/PandaLift/fail_rollout" \
  data.camera_name="agentview" \
  model.lpb_ckpt="${LPB_CKPT_PATH}" \
  feature.device="cuda" \
  detector.device="cuda" \
  detector.delta=10.0 \
  detector.delta_step=0.5 \
  detector.lambda_mode="mean" \
  detector.lambda_window_size=-1 \
  labels.fail_tail_ratio=0.2 \
  online.adaptive_delta="${ADAPTIVE_DELTA}" \
  save_dir="${ROOT_DIR}/checkpoints/PandaLift/discriminator/lpb_knn/contrastive" \
  "$@"
