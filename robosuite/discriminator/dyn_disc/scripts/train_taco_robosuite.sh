#!/usr/bin/env bash
# Train the TACO representation ablation on robosuite success rollouts.

set -euo pipefail

export HYDRA_FULL_ERROR=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

ARGS=(-m robosuite.discriminator.dyn_disc.training.train --config-name train_taco_robosuite "$@")

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node="${NPROC_PER_NODE}" \
        "${ARGS[@]}"
else
    "${PYTHON_BIN}" "${ARGS[@]}"
fi
