#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

SEED="${SEED:-42}"
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
DSM_CKPT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/lpb_score/lpb_score_dsm_20260407_211052/lpb_score_dsm_20260407_211052.pt"
SAVE_DIR="${SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_score/analyse}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[analyse_lpb_score_dsm] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

if [[ ! -f "${DSM_CKPT}" ]]; then
  echo "[analyse_lpb_score_dsm] Missing DSM checkpoint: ${DSM_CKPT}" >&2
  exit 1
fi

echo "[analyse_lpb_score_dsm] ROOT=${ROOT}"
echo "[analyse_lpb_score_dsm] policy.ckpt=${CKPT}"
echo "[analyse_lpb_score_dsm] model.dsm_ckpt=${DSM_CKPT}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_score/analyse.py" \
  seed="${SEED}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${CKPT}" \
  model.dsm_ckpt="${DSM_CKPT}" \
  "$@"
