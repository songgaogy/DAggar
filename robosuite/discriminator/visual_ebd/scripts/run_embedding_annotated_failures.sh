#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

GPU="${GPU:-0}"
SEED=3

EMBED_METHOD="tsne"   # pca, umap, tsne
SUCCESS_MAX_ROLLOUTS_PER_TASK=0  # 0: disable success rollout points, -1: use all

CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/outputs/embedding_annotated_failures}"
FAILURE_ROOT="${FAILURE_ROOT:-${ROOT}/data/utils/fail_rollout}"
FAILURE_MAX_ROLLOUTS_PER_TASK="${FAILURE_MAX_ROLLOUTS_PER_TASK:-0}"
MAX_TOTAL_POINTS="${MAX_TOTAL_POINTS:-30000}"
TIMESTEP_STRIDE="${TIMESTEP_STRIDE:-1}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[run_embedding_annotated_failures] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[run_embedding_annotated_failures] ROOT=${ROOT}"
echo "[run_embedding_annotated_failures] policy.ckpt=${CKPT}"
echo "[run_embedding_annotated_failures] failure_root=${FAILURE_ROOT}"
echo "[run_embedding_annotated_failures] embed_method=${EMBED_METHOD}"
echo "[run_embedding_annotated_failures] save_dir=${SAVE_DIR}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/tsne/analyse.py" \
  seed="${SEED}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${CKPT}" \
  embedding.method="${EMBED_METHOD}" \
  data.max_rollouts_per_task_source="${SUCCESS_MAX_ROLLOUTS_PER_TASK}" \
  data.timestep_stride="${TIMESTEP_STRIDE}" \
  analysis.max_total_points="${MAX_TOTAL_POINTS}" \
  annotated_failures.enabled=true \
  annotated_failures.root_dir="${FAILURE_ROOT}" \
  annotated_failures.max_rollouts_per_task="${FAILURE_MAX_ROLLOUTS_PER_TASK}" \
  annotated_failures.max_success_rollouts_per_task="${SUCCESS_MAX_ROLLOUTS_PER_TASK}" \
  "$@"
