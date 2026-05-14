#!/usr/bin/env bash
# LPB v2 latent visualization entry point for real-world Agilex benchmark data.
# Typical usage:
#   BENCHMARK_JSON=checkpoints/lpb_v2/real_world_eval/run_20260429_103014/benchmark.json \
#     bash robosuite/discriminator/lpb_v2/scripts/vis_realworld_latent.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/agilex/failure_annotations/out_by_task}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/agilex}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/data/.agilex_train_cache}"
TASKS="${TASKS:-candy_in_plate duck_in_bowl Micky_in_box sausage_in_pot}"

MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-25}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-25}"

EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/checkpoints/lpb_v2/real_world_eval}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-run_20260429_103014}"
BENCHMARK_JSON="${BENCHMARK_JSON:-${EVAL_ROOT}/${SOURCE_RUN_NAME}/benchmark.json}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/${RUN_NAME}_vis-dinov3}"
mkdir -p "${OUT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-lpb-v2-latent-${RUN_NAME}}"
mkdir -p "${MPLCONFIGDIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# Right-arm Agilex defaults: qpos[:, 7:14], action[:, 7:14].
ACTION_START="${ACTION_START:-7}"
ACTION_STOP="${ACTION_STOP:-14}"
PROPRIO_FIELD="${PROPRIO_FIELD:-qpos}"
PROPRIO_START="${PROPRIO_START:-7}"
PROPRIO_STOP="${PROPRIO_STOP:-14}"

DEVICE="${DEVICE:-cuda}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-32}"

# --------------------------------------------------------
# auto detect normalizers
MODEL_CKPT="${MODEL_CKPT:-checkpoints/lpb_v2/dynamics/agilex_train-dinov3-20260513_221529/checkpoints/model_39.pth}"

KNN_FEATURE_SOURCE="${KNN_FEATURE_SOURCE:-transformer}"
KNN_TRANSFORMER_LAYER="${KNN_TRANSFORMER_LAYER:-1}"
SEED="${SEED:-0}"
# --------------------------------------------------------


TSNE_MAX_POINTS="${TSNE_MAX_POINTS:-20000}"
TSNE_PERPLEXITY="${TSNE_PERPLEXITY:-30.0}"
TSNE_LEARNING_RATE="${TSNE_LEARNING_RATE:-auto}"
TSNE_INIT="${TSNE_INIT:-pca}"
TSNE_ITERATIONS="${TSNE_ITERATIONS:-1000}"
POINT_SIZE="${POINT_SIZE:-4.0}"
ALPHA="${ALPHA:-0.65}"

EXTRA_ARGS=()
if [[ "${USE_CACHE:-1}" == "1" ]]; then
    EXTRA_ARGS+=(--cache-root "${CACHE_ROOT}")
fi
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
if [[ -n "${BENCHMARK_JSON:-}" ]]; then
    EXTRA_ARGS+=(--benchmark-json "${BENCHMARK_JSON}")
fi

"${PYTHON_BIN}" -m robosuite.discriminator.lpb_v2.utils.vis_latent \
    --model-ckpt            "${MODEL_CKPT}" \
    --fail-root             "${FAIL_ROOT}" \
    --success-root          "${SUCCESS_ROOT}" \
    --tasks                 ${TASKS} \
    --out-dir               "${OUT_DIR}" \
    --proprio-field         "${PROPRIO_FIELD}" \
    --proprio-start         "${PROPRIO_START}" \
    --proprio-stop          "${PROPRIO_STOP}" \
    --action-start          "${ACTION_START}" \
    --action-stop           "${ACTION_STOP}" \
    --device                "${DEVICE}" \
    --encode-batch-size     "${ENCODE_BATCH_SIZE}" \
    --knn-feature-source    "${KNN_FEATURE_SOURCE}" \
    --knn-transformer-layer "${KNN_TRANSFORMER_LAYER}" \
    --seed                  "${SEED}" \
    --tsne-max-points       "${TSNE_MAX_POINTS}" \
    --tsne-perplexity       "${TSNE_PERPLEXITY}" \
    --tsne-learning-rate    "${TSNE_LEARNING_RATE}" \
    --tsne-init             "${TSNE_INIT}" \
    --tsne-iterations       "${TSNE_ITERATIONS}" \
    --point-size            "${POINT_SIZE}" \
    --alpha                 "${ALPHA}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[real_world][lpb_v2][latent] wrote to ${OUT_DIR}"
