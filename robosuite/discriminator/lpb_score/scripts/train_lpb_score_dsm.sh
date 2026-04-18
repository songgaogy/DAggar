#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

GPU="${GPU:-0}"
SEED="${SEED:-42}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_dipole-new-v4}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="lpb_dipole_dsm_${TIMESTAMP}"
SAVE_NAME="${RUN_NAME}.pt"
SAVE_DIR="${BASE_SAVE_DIR}/${RUN_NAME}"
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"

BATCH_SIZE=128
NUM_WORKERS=16
PREFETCH_FACTOR=8
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
PIN_MEMORY="${PIN_MEMORY:-1}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"
NUM_POS=100
NUM_NEG=100
POSITIVE_RATIO="${POSITIVE_RATIO:-0.7}"
EPOCHS="${EPOCHS:-25}"

LR="${LR:-2e-4}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
HORIZON="${HORIZON:-10}"
DSM_WINDOW_SIZE="${DSM_WINDOW_SIZE:-${HORIZON}}"
ENCODER_BATCH_SIZE="${ENCODER_BATCH_SIZE:-64}"
TRAIN_ENCODER="${TRAIN_ENCODER:-1}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
PREPROCESSED_CACHE_DIR="${PREPROCESSED_CACHE_DIR:-${ROOT}/data/.lpb_score_preprocessed_cache}"
USE_PREPROCESSED_CACHE="${USE_PREPROCESSED_CACHE:-1}"
REFRESH_PREPROCESSED_CACHE="${REFRESH_PREPROCESSED_CACHE:-0}"
STD_CLAMP_MIN="${STD_CLAMP_MIN:-0.05}"
NOISE_SCALE="${NOISE_SCALE:-0.08}"
TEMPORAL_KERNEL_SIZE="${TEMPORAL_KERNEL_SIZE:-3}"

if [[ "${TRAIN_ENCODER}" == "1" ]]; then
  TRAIN_ENCODER_BOOL="true"
else
  TRAIN_ENCODER_BOOL="false"
fi

if [[ "${USE_PREPROCESSED_CACHE}" == "1" ]]; then
  USE_PREPROCESSED_CACHE_BOOL="true"
else
  USE_PREPROCESSED_CACHE_BOOL="false"
fi

if [[ "${REFRESH_PREPROCESSED_CACHE}" == "1" ]]; then
  REFRESH_PREPROCESSED_CACHE_BOOL="true"
else
  REFRESH_PREPROCESSED_CACHE_BOOL="false"
fi

if [[ "${PERSISTENT_WORKERS}" == "1" ]]; then
  PERSISTENT_WORKERS_BOOL="true"
else
  PERSISTENT_WORKERS_BOOL="false"
fi

if [[ "${PIN_MEMORY}" == "1" ]]; then
  PIN_MEMORY_BOOL="true"
else
  PIN_MEMORY_BOOL="false"
fi

if [[ "${CUDNN_BENCHMARK}" == "1" ]]; then
  CUDNN_BENCHMARK_BOOL="true"
else
  CUDNN_BENCHMARK_BOOL="false"
fi


if [[ ! -f "${CKPT}" ]]; then
  echo "[train_lpb_score_dsm] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[train_lpb_score_dsm] ROOT=${ROOT}"
echo "[train_lpb_score_dsm] policy.ckpt=${CKPT}"
echo "[train_lpb_score_dsm] save_dir=${SAVE_DIR}"
echo "[train_lpb_score_dsm] train.num_pos_traj=${NUM_POS}"
echo "[train_lpb_score_dsm] train.num_neg_traj=${NUM_NEG}"
echo "[train_lpb_score_dsm] dataset.window_size=${DSM_WINDOW_SIZE}"
echo "[train_lpb_score_dsm] training.batch_size=${BATCH_SIZE} policy.encoder_batch_size=${ENCODER_BATCH_SIZE} training.num_workers=${NUM_WORKERS}"
echo "[train_lpb_score_dsm] training.prefetch_factor=${PREFETCH_FACTOR} training.persistent_workers=${PERSISTENT_WORKERS_BOOL} training.pin_memory=${PIN_MEMORY_BOOL}"
echo "[train_lpb_score_dsm] training.cudnn_benchmark=${CUDNN_BENCHMARK_BOOL}"
echo "[train_lpb_score_dsm] policy.trainable_encoder=${TRAIN_ENCODER_BOOL}"
echo "[train_lpb_score_dsm] policy.lora={enabled=${TRAIN_ENCODER_BOOL}, rank=${LORA_RANK}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}}"
echo "[train_lpb_score_dsm] model.kernel_size=${TEMPORAL_KERNEL_SIZE}"
echo "[train_lpb_score_dsm] data.preprocessed_cache_dir=${PREPROCESSED_CACHE_DIR}"
echo "[train_lpb_score_dsm] data.use_preprocessed_cache=${USE_PREPROCESSED_CACHE_BOOL}"
echo "[train_lpb_score_dsm] data.refresh_preprocessed_cache=${REFRESH_PREPROCESSED_CACHE_BOOL}"

"${PYTHON_BIN}" "${ROOT}/robosuite/discriminator/lpb_score/train.py" \
  seed="${SEED}" \
  hydra.run.dir="${SAVE_DIR}" \
  save_name="${SAVE_NAME}" \
  save_dir="${SAVE_DIR}" \
  policy.ckpt="${CKPT}" \
  policy.encoder_batch_size="${ENCODER_BATCH_SIZE}" \
  policy.trainable_encoder="${TRAIN_ENCODER_BOOL}" \
  policy.lora.enabled="${TRAIN_ENCODER_BOOL}" \
  policy.lora.rank="${LORA_RANK}" \
  policy.lora.alpha="${LORA_ALPHA}" \
  policy.lora.dropout="${LORA_DROPOUT}" \
  data.image_size="${IMAGE_SIZE}" \
  data.preprocessed_cache_dir="${PREPROCESSED_CACHE_DIR}" \
  data.use_preprocessed_cache="${USE_PREPROCESSED_CACHE_BOOL}" \
  data.refresh_preprocessed_cache="${REFRESH_PREPROCESSED_CACHE_BOOL}" \
  dataset.window_size="${DSM_WINDOW_SIZE}" \
  model.kernel_size="${TEMPORAL_KERNEL_SIZE}" \
  model.std_clamp_min="${STD_CLAMP_MIN}" \
  model.noise_scale="${NOISE_SCALE}" \
  data.splits.train.num_pos_traj="${NUM_POS}" \
  data.splits.train.num_neg_traj="${NUM_NEG}" \
  training.batch_size="${BATCH_SIZE}" \
  training.num_workers="${NUM_WORKERS}" \
  training.prefetch_factor="${PREFETCH_FACTOR}" \
  training.persistent_workers="${PERSISTENT_WORKERS_BOOL}" \
  training.pin_memory="${PIN_MEMORY_BOOL}" \
  training.cudnn_benchmark="${CUDNN_BENCHMARK_BOOL}" \
  training.epochs="${EPOCHS}" \
  training.lr="${LR}" \
  training.positive_ratio="${POSITIVE_RATIO}" \
  "$@"
