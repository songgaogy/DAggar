#!/usr/bin/env bash
# Launch one controlled positive-policy ablation condition (C0-C5).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
CONDITION="${CONDITION:-${1:-}}"
if [[ -z "${CONDITION}" ]]; then
  echo "[ERROR] Set CONDITION=C0..C5 or pass it as the first argument." >&2
  exit 2
fi

case "${CONDITION^^}" in
  C0) MODE=coupled;         INCLUDE_SUCCESS=false ;;
  C1) MODE=coupled;         INCLUDE_SUCCESS=true  ;;
  C2) MODE=independent_soft; INCLUDE_SUCCESS=false ;;
  C3) MODE=independent_soft; INCLUDE_SUCCESS=true  ;;
  C4) MODE=filtered_bc;      INCLUDE_SUCCESS=false ;;
  C5) MODE=filtered_bc;      INCLUDE_SUCCESS=true  ;;
  *)
    echo "[ERROR] Unknown CONDITION=${CONDITION}; expected C0..C5." >&2
    exit 2
    ;;
esac

CONDITION="${CONDITION^^}"
MATRIX_ROOT="${MATRIX_ROOT:-outputs/debug/offline_disc_positive_matrix}"
export ROOT_DIR
export POSITIVE_TRAINING_MODE="${MODE}"
export INCLUDE_PURE_SUCCESS="${INCLUDE_SUCCESS}"
export STAGE_RUN_DIR="${MATRIX_ROOT}/${CONDITION}"
export SEED="${SEED:-42}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-15000}"
export POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-256}"
export BRANCH_BETA="${BRANCH_BETA:-1}"

echo "[positive_matrix] condition=${CONDITION} mode=${MODE} include_pure_success=${INCLUDE_SUCCESS}"
EXTRA_OVERRIDES=()
if [[ "${CONDITION}" == "C5" ]]; then
  EXTRA_OVERRIDES+=(
    "offline.positive_training.expected_human_transitions=1892"
    "offline.positive_training.expected_pretrain_transitions=2360"
    "offline.positive_training.expected_pure_success_transitions=3270"
    "offline.positive_training.expected_total_positive_transitions=7522"
  )
fi
exec bash "${ROOT_DIR}/robosuite/pipeline/offline/scripts/train_offline_dipole.sh" \
  "${EXTRA_OVERRIDES[@]}"
