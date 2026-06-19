#!/usr/bin/env bash
# Train WAM-style latent dynamics on robosuite with a frozen DINOv3 encoder.
# Run from repo root:
#   bash robosuite/discriminator/dyn_disc/scripts/train_dinov3_robosuite_dynamics.sh
# Multi-GPU:
#   NPROC_PER_NODE=2 bash robosuite/discriminator/dyn_disc/scripts/train_dinov3_robosuite_dynamics.sh

set -euo pipefail

export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_NAME="${WANDB_NAME:-songgao-personal}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
NPROC_PER_NODE=2

ARGS=(-m robosuite.discriminator.dyn_disc.training.train --config-name train_dinov3_dyn_robosuite "$@")

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" "${ARGS[@]}"
else
    "${PYTHON_BIN}" "${ARGS[@]}"
fi
