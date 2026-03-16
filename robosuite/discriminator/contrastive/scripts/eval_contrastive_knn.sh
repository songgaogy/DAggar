#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/dodo/Documents/DAggar/robosuite"
PYTHON_BIN="/home/dodo/miniconda3/envs/daggar/bin/python"

export CUDA_VISIBLE_DEVICES=0

PURE_CONTRASTIVE_CKPT_PATH="${PURE_CONTRASTIVE_CKPT_PATH:-${ROOT_DIR}/checkpoints/PandaLift/discriminator/pure_contrastive}"
WINDOW_LEN=10
ADAPTIVE_DELTA=true

cd "${ROOT_DIR}"

exec "${PYTHON_BIN}" -m robosuite.discriminator.contrastive.eval_knn_discriminator \
  data.expert_dir="${ROOT_DIR}/data/PandaLift/expert" \
  data.fail_rollout_dir="${ROOT_DIR}/data/PandaLift/fail_rollout" \
  data.camera_name="agentview" \
  model.checkpoint_path="${PURE_CONTRASTIVE_CKPT_PATH}" \
  feature.device="cuda" \
  detector.device="cuda" \
  detector.delta=10.0 \
  detector.delta_step=0.5 \
  detector.lambda_mode="mean" \
  detector.lambda_window_size=$WINDOW_LEN \
  labels.fail_tail_ratio=0.3 \
  online.adaptive_delta=$ADAPTIVE_DELTA \
  save_dir="${ROOT_DIR}/checkpoints/PandaLift/discriminator/pure_contrastive_knn" \
  "$@"
