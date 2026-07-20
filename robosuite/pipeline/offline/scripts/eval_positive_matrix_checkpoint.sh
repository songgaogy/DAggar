#!/usr/bin/env bash
# Evaluate one positive-policy matrix checkpoint at omega=0.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
CONDITION="${CONDITION:-${1:-}}"
COMPLETED_STEPS="${COMPLETED_STEPS:-${2:-}}"
if [[ -z "${CONDITION}" || -z "${COMPLETED_STEPS}" ]]; then
  echo "[ERROR] Usage: CONDITION=C0 COMPLETED_STEPS=5000 $0" >&2
  exit 2
fi

MATRIX_ROOT="${MATRIX_ROOT:-outputs/debug/offline_disc_positive_matrix}"
RUN_DIR="${MATRIX_ROOT}/${CONDITION^^}"
CHECKPOINT="${RUN_DIR}/checkpoints/step_$(printf '%08d' "${COMPLETED_STEPS}").pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if (( COMPLETED_STEPS >= 15000 )); then
  DEFAULT_EPISODES=50
else
  DEFAULT_EPISODES=20
fi

export ROOT_DIR
export POLICY_CKPT="${CHECKPOINT}"
export OMEGAS=0
export EVAL_EPISODES="${EVAL_EPISODES:-${DEFAULT_EPISODES}}"
export EVAL_EPISODE_MAX_STEPS="${EVAL_EPISODE_MAX_STEPS:-500}"
export SAVE_VIDEO="${SAVE_VIDEO:-false}"
export SEED="${SEED:-10086}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_DIR}/eval_matrix}"

exec bash "${ROOT_DIR}/robosuite/pipeline/offline/scripts/eval_offline_dipole.sh"
