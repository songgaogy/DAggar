#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

SEED="${SEED:-42}"
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
LPB_CKPT="${LPB_CKPT:-${ROOT}/checkpoints/multitask_6/lpb_new/lpb_new_20260324_232921/lpb_new_20260324_232921.pt}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_new/analyse}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[analyse_lpb_new] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

if [[ ! -f "${LPB_CKPT}" ]]; then
  echo "[analyse_lpb_new] Missing LPB checkpoint: ${LPB_CKPT}" >&2
  exit 1
fi

echo "[analyse_lpb_new] ROOT=${ROOT}"
echo "[analyse_lpb_new] policy.ckpt=${CKPT}"
echo "[analyse_lpb_new] model.lpb_ckpt=${LPB_CKPT}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_new/analyse.py" \
  seed="${SEED}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${CKPT}" \
  model.lpb_ckpt="${LPB_CKPT}" \
  "$@"
