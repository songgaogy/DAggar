#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
TPUD_CKPT="checkpoints/multitask_6/tpud/tpud_20260324_211502/tpud_20260324_211502_ep0050.pt"
SAVE_DIR="${SAVE_DIR:-${ROOT}/checkpoints/multitask_6/tpud/visualize}"

export CUDA_VISIBLE_DEVICES=1

cd "${ROOT}"
"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/bce/visualize_failures.py" \
  policy.ckpt="${CKPT}" \
  model.tpud_ckpt="${TPUD_CKPT}" \
  save_dir="${SAVE_DIR}" \
  visualization.num_videos=10 \
  "$@"
