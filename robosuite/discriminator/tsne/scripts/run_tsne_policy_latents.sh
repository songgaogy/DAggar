#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/outputs/tsne}"
MAX_ROLLOUTS_PER_TASK_SOURCE="${MAX_ROLLOUTS_PER_TASK_SOURCE:-20}"
MAX_TOTAL_POINTS="${MAX_TOTAL_POINTS:-30000}"
TIMESTEP_STRIDE="${TIMESTEP_STRIDE:-1}"
INCLUDE_SUBOPTIMAL="${INCLUDE_SUBOPTIMAL:-true}"
SUBOPTIMAL_MAX_ROLLOUTS_PER_TASK="${SUBOPTIMAL_MAX_ROLLOUTS_PER_TASK:-8}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[run_tsne_policy_latents] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[run_tsne_policy_latents] ROOT=${ROOT}"
echo "[run_tsne_policy_latents] policy.ckpt=${CKPT}"
echo "[run_tsne_policy_latents] save_dir=${SAVE_DIR}"
echo "[run_tsne_policy_latents] include_suboptimal=${INCLUDE_SUBOPTIMAL}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/tsne/analyse.py" \
  seed="${SEED}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${CKPT}" \
  data.max_rollouts_per_task_source="${MAX_ROLLOUTS_PER_TASK_SOURCE}" \
  data.timestep_stride="${TIMESTEP_STRIDE}" \
  analysis.max_total_points="${MAX_TOTAL_POINTS}" \
  suboptimal.enabled="${INCLUDE_SUBOPTIMAL}" \
  suboptimal.max_rollouts_per_task="${SUBOPTIMAL_MAX_ROLLOUTS_PER_TASK}" \
  "$@"
