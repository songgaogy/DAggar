#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 TASK CONFIG PHYSICAL_GPU_ID" >&2
    exit 2
fi

TASK="$1"
CONFIG="$2"
GPU_ID="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SWEEP_ROOT="${SWEEP_ROOT:-${REPO_ROOT}/checkpoints/dyn_disc/pu_bce_eval_robosuite-chunk/trick_sweep_20260716_v1}"
OUT_DIR="${SWEEP_ROOT}/${TASK}/${CONFIG}/seed0"

THRESHOLD_NORMALIZATION="none"
SOFT_CAP_C=""
SOFT_CAP_LAMBDA="0.0"
case "${CONFIG}" in
    baseline)
        ;;
    norm_only)
        THRESHOLD_NORMALIZATION="epoch_boundary"
        ;;
    cap_c5_l1e3)
        SOFT_CAP_C="5"; SOFT_CAP_LAMBDA="1e-3"
        ;;
    cap_c5_l1e2)
        SOFT_CAP_C="5"; SOFT_CAP_LAMBDA="1e-2"
        ;;
    cap_c8_l1e3)
        SOFT_CAP_C="8"; SOFT_CAP_LAMBDA="1e-3"
        ;;
    cap_c8_l1e2)
        SOFT_CAP_C="8"; SOFT_CAP_LAMBDA="1e-2"
        ;;
    norm_cap_c5_l1e3)
        THRESHOLD_NORMALIZATION="epoch_boundary"; SOFT_CAP_C="5"; SOFT_CAP_LAMBDA="1e-3"
        ;;
    norm_cap_c5_l1e2)
        THRESHOLD_NORMALIZATION="epoch_boundary"; SOFT_CAP_C="5"; SOFT_CAP_LAMBDA="1e-2"
        ;;
    norm_cap_c8_l1e3)
        THRESHOLD_NORMALIZATION="epoch_boundary"; SOFT_CAP_C="8"; SOFT_CAP_LAMBDA="1e-3"
        ;;
    norm_cap_c8_l1e2)
        THRESHOLD_NORMALIZATION="epoch_boundary"; SOFT_CAP_C="8"; SOFT_CAP_LAMBDA="1e-2"
        ;;
    *)
        echo "Unknown config: ${CONFIG}" >&2
        exit 2
        ;;
esac

if [[ -e "${OUT_DIR}/trial_summary.json" ]]; then
    echo "Refusing to overwrite completed trial: ${OUT_DIR}" >&2
    exit 3
fi
mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
TASKS="${TASK}" \
CONFIG_NAME="${CONFIG}" \
THRESHOLD_NORMALIZATION="${THRESHOLD_NORMALIZATION}" \
SOFT_CAP_C="${SOFT_CAP_C}" \
SOFT_CAP_LAMBDA="${SOFT_CAP_LAMBDA}" \
SOFT_CAP_TEMPERATURE="1.0" \
CHECKPOINT_EPOCHS="1 2 5 10 20" \
EPOCHS="20" \
SEED="0" \
OUT_DIR="${OUT_DIR}" \
SAVE_JSON="${OUT_DIR}/benchmark.json" \
SAVE_CKPT_DIR="${OUT_DIR}/checkpoints" \
TENSORBOARD_DIR="${OUT_DIR}/tensorboard" \
bash "${REPO_ROOT}/robosuite/discriminator/dyn_disc/scripts/run_pu_bce_robosuite_benchmark.sh" \
    2>&1 | tee "${OUT_DIR}/run.log"
