#!/usr/bin/env bash
# Omega sweep on a single trained D4-Disc checkpoint.
# One trained model serves all omega — no retraining required.
# Usage (from repo root):
#   D4_CKPT=/abs/path/d4_dynamics.pt \
#     bash robosuite/discriminator/d4disc/scripts/sweep_omega_d4.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

D4_CKPT="${D4_CKPT:?must set D4_CKPT}"
OMEGAS="${OMEGAS:-0 0.1 0.2 0.5 1 2}"

RUN_TAG="${RUN_TAG:-omega_sweep_$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="${SWEEP_DIR:-${REPO_ROOT}/checkpoints/d4disc/eval/${RUN_TAG}}"
mkdir -p "${SWEEP_DIR}"

for OMEGA in ${OMEGAS}; do
    SAFE_OMEGA="$(printf '%s' "${OMEGA}" | tr '.' 'p')"
    SUBRUN="w${SAFE_OMEGA}"
    OUT_DIR="${SWEEP_DIR}/${SUBRUN}"
    SAVE_JSON="${OUT_DIR}/benchmark.json"
    mkdir -p "${OUT_DIR}"
    echo "[sweep_omega] omega=${OMEGA} -> ${SAVE_JSON}"
    OMEGA="${OMEGA}" D4_CKPT="${D4_CKPT}" \
        OUT_DIR="${OUT_DIR}" SAVE_JSON="${SAVE_JSON}" RUN_NAME="${SUBRUN}" \
        bash "${REPO_ROOT}/robosuite/discriminator/d4disc/scripts/run_d4_benchmark.sh" \
        "$@"
done

echo "[sweep_omega] wrote results under ${SWEEP_DIR}"
