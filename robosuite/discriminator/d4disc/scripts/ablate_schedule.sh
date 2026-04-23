#!/usr/bin/env bash
# Schedule ablation for D4-Disc: sweep (alpha_exponent x alpha_cap_rate).
# Each point is a separate training run -- this script spawns many training
# jobs sequentially. Keep the grid small.
#   ALPHA_EXPONENTS="0.5 0.75 1.0" ALPHA_CAP_RATES="0.5 1.0 2.0" \
#     bash robosuite/discriminator/d4disc/scripts/ablate_schedule.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

ALPHA_EXPONENTS="${ALPHA_EXPONENTS:-0.5 0.75 1.0}"
ALPHA_CAP_RATES="${ALPHA_CAP_RATES:-0.5 1.0 2.0}"

ABL_TAG="${ABL_TAG:-ablate_schedule_$(date +%Y%m%d_%H%M%S)}"
ABL_DIR="${ABL_DIR:-${REPO_ROOT}/checkpoints/d4disc/dynamics/${ABL_TAG}}"
mkdir -p "${ABL_DIR}"

for AE in ${ALPHA_EXPONENTS}; do
    for AC in ${ALPHA_CAP_RATES}; do
        SAFE_AE="$(printf '%s' "${AE}" | tr '.' 'p')"
        SAFE_AC="$(printf '%s' "${AC}" | tr '.' 'p')"
        SUB="ae${SAFE_AE}_ac${SAFE_AC}"
        SAVE_DIR="${ABL_DIR}/${SUB}"
        mkdir -p "${SAVE_DIR}"
        echo "[ablate] alpha_exponent=${AE} alpha_cap_rate=${AC} -> ${SAVE_DIR}"
        ALPHA_EXPONENT="${AE}" ALPHA_CAP_RATE="${AC}" \
            SAVE_DIR="${SAVE_DIR}" RUN_NAME="${SUB}" \
            bash "${REPO_ROOT}/robosuite/discriminator/d4disc/scripts/train_d4.sh" \
            "$@"
    done
done

echo "[ablate] wrote checkpoints under ${ABL_DIR}"
