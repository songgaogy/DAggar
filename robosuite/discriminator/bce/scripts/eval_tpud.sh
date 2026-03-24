#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

GPU="${GPU:-0}"
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
TPUD_CKPT="${TPUD_CKPT:-${ROOT}/checkpoints/multitask_6/tpud/tpud.pt}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/checkpoints/multitask_6/tpud/eval}"

export CUDA_VISIBLE_DEVICES="${GPU}"

cd "${ROOT}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/bce/eval_tpud_discriminator.py" \
  policy.ckpt="${CKPT}" \
  model.tpud_ckpt="${TPUD_CKPT}" \
  save_dir="${SAVE_DIR}" \
  "$@"
