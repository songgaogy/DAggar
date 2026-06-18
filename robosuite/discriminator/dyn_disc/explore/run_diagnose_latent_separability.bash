#!/usr/bin/env bash
# LPB v2 Phase-1 latent-separability diagnostic for real-world AgileX features.
# Pooled-across-tasks MMD + Ledoit-Wolf Mahalanobis AUROC + matched-timestep
# AUROC on the WAM transformer-layer-1 latent.
#
# Typical usage:
#   MODEL_CKPT=checkpoints/dyn_disc/dynamics/agilex_train-20260429_003751/checkpoints/model_49.pth \
#     bash robosuite/discriminator/dyn_disc/explore/run_diagnose_latent_separability.bash
#
# To re-analyze a previously cached latents.npz without re-encoding:
#   LOAD_CACHE=/path/to/latents.npz \
#     bash robosuite/discriminator/dyn_disc/explore/run_diagnose_latent_separability.bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/agilex/failure_annotations/out_by_task}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/agilex}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/data/.agilex_train_cache}"
TASKS="${TASKS:-candy_in_plate duck_in_bowl Micky_in_box sausage_in_pot}"

MODEL_CKPT="${MODEL_CKPT:-checkpoints/dyn_disc/dynamics/agilex_train-20260429_003751/checkpoints/model_49.pth}"
LOAD_CACHE="${LOAD_CACHE:-}"
CACHE_FEATURES_ONLY="${CACHE_FEATURES_ONLY:-0}"

if [[ -z "${LOAD_CACHE}" && ! -f "${MODEL_CKPT}" ]]; then
    echo "[dyn_disc][sep] ERROR: MODEL_CKPT not found: ${MODEL_CKPT}" >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/checkpoints/dyn_disc/latent_separability}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/${RUN_NAME}}"
mkdir -p "${OUT_DIR}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_NAME="${WANDB_NAME:-songgao-personal}"

MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-50}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-100}"
USE_CACHE="${USE_CACHE:-1}"

PROPRIO_FIELD="${PROPRIO_FIELD:-qpos}"
PROPRIO_START="${PROPRIO_START:-7}"
PROPRIO_STOP="${PROPRIO_STOP:-14}"
ACTION_START="${ACTION_START:-7}"
ACTION_STOP="${ACTION_STOP:-14}"

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"
FEATURE_SOURCE="${FEATURE_SOURCE:-transformer}"
TRANSFORMER_LAYER="${TRANSFORMER_LAYER:-1}"

SEED="${SEED:-0}"
PERMUTATIONS="${PERMUTATIONS:-1000}"
MAX_FRAMES_PER_CLASS="${MAX_FRAMES_PER_CLASS:-4000}"
MEDIAN_MAX_POINTS="${MEDIAN_MAX_POINTS:-2000}"
MAHALANOBIS_TRAIN_FRAC="${MAHALANOBIS_TRAIN_FRAC:-0.8}"
MATCHED_WINDOW="${MATCHED_WINDOW:-5}"

EXTRA_ARGS=()
if [[ -n "${LOAD_CACHE}" ]]; then
    EXTRA_ARGS+=(--load-cache "${LOAD_CACHE}")
else
    EXTRA_ARGS+=(--model-ckpt "${MODEL_CKPT}")
    EXTRA_ARGS+=(--fail-root "${FAIL_ROOT}")
    EXTRA_ARGS+=(--success-root "${SUCCESS_ROOT}")
    if [[ "${USE_CACHE}" == "1" ]]; then
        EXTRA_ARGS+=(--cache-root "${CACHE_ROOT}")
    fi
    EXTRA_ARGS+=(--tasks ${TASKS})
    EXTRA_ARGS+=(--proprio-field "${PROPRIO_FIELD}")
    EXTRA_ARGS+=(--proprio-start "${PROPRIO_START}")
    EXTRA_ARGS+=(--proprio-stop  "${PROPRIO_STOP}")
    EXTRA_ARGS+=(--action-start  "${ACTION_START}")
    EXTRA_ARGS+=(--action-stop   "${ACTION_STOP}")
    EXTRA_ARGS+=(--device        "${DEVICE}")
    EXTRA_ARGS+=(--encode-batch-size "${ENCODE_BATCH_SIZE}")
    if [[ -n "${MAX_FAIL_PER_TASK}" && "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
        EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
    fi
    if [[ -n "${MAX_SUCCESS_PER_TASK}" && "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
        EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
    fi
    if [[ -n "${PROPRIO_INDICES:-}" ]]; then
        EXTRA_ARGS+=(--proprio-indices ${PROPRIO_INDICES})
    fi
    if [[ -n "${CAMERA_TO_VIEW:-}" ]]; then
        EXTRA_ARGS+=(--camera-to-view "${CAMERA_TO_VIEW}")
    fi
fi

if [[ "${CACHE_FEATURES_ONLY}" == "1" ]]; then
    EXTRA_ARGS+=(--cache-features-only)
fi

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.explore.diagnose_latent_separability \
    --out-dir                        "${OUT_DIR}" \
    --feature-source                 "${FEATURE_SOURCE}" \
    --transformer-layer              "${TRANSFORMER_LAYER}" \
    --seed                           "${SEED}" \
    --permutations                   "${PERMUTATIONS}" \
    --max-frames-per-class           "${MAX_FRAMES_PER_CLASS}" \
    --median-max-points              "${MEDIAN_MAX_POINTS}" \
    --mahalanobis-train-frac         "${MAHALANOBIS_TRAIN_FRAC}" \
    --matched-window                 "${MATCHED_WINDOW}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[dyn_disc][sep] wrote outputs to ${OUT_DIR}"
