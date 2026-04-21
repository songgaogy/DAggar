#!/usr/bin/env bash
# Ablation sweep over (omega, k, beta_scale) for D3-Disc.
# Runs the full benchmark (fit + evaluate) once per configuration and writes
# benchmark.json under checkpoints/d3disc/eval/abl_w<omega>_k<k>_b<scale>/.
#
# User-facing knobs forwarded verbatim to run_d3_benchmark.sh (SUCC_NUM / FAIL_NUM / TASKS / etc.).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${HERE}/run_d3_benchmark.sh"

# Sweep axes. Override via env vars: OMEGAS, KS, BETA_SCALES.
OMEGAS="${OMEGAS:-0 0.1 0.2 0.5 1}"
KS="${KS:-1 5}"
BETA_SCALES="${BETA_SCALES:-auto}"  # only 'auto' by default — pass multiple to scale 1/MAD.

for OMEGA in ${OMEGAS}; do
    for K in ${KS}; do
        for BETA_SCALE in ${BETA_SCALES}; do
            RUN_NAME="abl_w${OMEGA}_k${K}_b${BETA_SCALE}"
            BETA_VALUE="${BETA_SCALE}"
            echo "==== [abl] ${RUN_NAME}  ===="
            OMEGA="${OMEGA}" K="${K}" BETA="${BETA_VALUE}" RUN_NAME="${RUN_NAME}" \
                bash "${RUN_SCRIPT}" "$@"
        done
    done
done
