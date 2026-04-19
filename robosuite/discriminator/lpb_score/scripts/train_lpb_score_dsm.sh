#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/dodo/miniconda3/envs/daggar/bin/torchrun}"

GPU="${GPU:-0,1}"
SEED="${SEED:-42}"
BASE_SAVE_DIR="${BASE_SAVE_DIR:-${ROOT}/checkpoints/multitask_6/lpb_dipole-new-v4}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-lpb_dipole_dsm_${TIMESTAMP}}"
SAVE_NAME="${SAVE_NAME:-${RUN_NAME}.pt}"
SAVE_DIR="${SAVE_DIR:-${BASE_SAVE_DIR}/${RUN_NAME}}"
CKPT="${CKPT:-${ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"

BATCH_SIZE=256
ENCODER_BATCH_SIZE=64
NUM_WORKERS=4
EPOCHS=25
TRAIN_ENCODER=0   # whether to train the encoder using LoRA
LORA_RANK=8
LORA_ALPHA=16.0

CUDA_PREFETCH=1
DATA_IN_RAM=100
TRAIN_PRELOAD_RAM_GB=16
VAL_PRELOAD_RAM_GB="${VAL_PRELOAD_RAM_GB:-0}"
LR="${LR:-2e-4}"
BATCH_PREFETCH_DEPTH="${BATCH_PREFETCH_DEPTH:-2}"
BATCHED_TRAJECTORY_CACHE_GB=16
PREPROCESSED_CACHE_LAYOUT="${PREPROCESSED_CACHE_LAYOUT:-bundle}"
UPGRADE_LEGACY_PREPROCESSED_CACHE="${UPGRADE_LEGACY_PREPROCESSED_CACHE:-1}"

NUM_POS=100
NUM_NEG=100
POSITIVE_RATIO="${POSITIVE_RATIO:-0.7}"


HORIZON="${HORIZON:-10}"
DSM_WINDOW_SIZE="${DSM_WINDOW_SIZE:-${HORIZON}}"

LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
DDP_FIND_UNUSED="${DDP_FIND_UNUSED:-0}"
DDP_STATIC_GRAPH="${DDP_STATIC_GRAPH:-1}"
DDP_BUCKET_VIEW="${DDP_BUCKET_VIEW:-1}"
USE_TMUX="${USE_TMUX:-1}"
TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/train.log}"

if [[ "${TRAIN_ENCODER}" == "1" ]]; then
  TRAIN_ENCODER_BOOL="true"
else
  TRAIN_ENCODER_BOOL="false"
fi

if [[ "${CUDA_PREFETCH}" == "1" ]]; then
  CUDA_PREFETCH_BOOL="true"
else
  CUDA_PREFETCH_BOOL="false"
fi

if [[ "${UPGRADE_LEGACY_PREPROCESSED_CACHE}" == "1" ]]; then
  UPGRADE_LEGACY_PREPROCESSED_CACHE_BOOL="true"
else
  UPGRADE_LEGACY_PREPROCESSED_CACHE_BOOL="false"
fi

if [[ "${DDP_FIND_UNUSED}" == "1" ]]; then
  DDP_FIND_UNUSED_BOOL="true"
else
  DDP_FIND_UNUSED_BOOL="false"
fi

if [[ "${DDP_STATIC_GRAPH}" == "1" ]]; then
  DDP_STATIC_GRAPH_BOOL="true"
else
  DDP_STATIC_GRAPH_BOOL="false"
fi

if [[ "${DDP_BUCKET_VIEW}" == "1" ]]; then
  DDP_BUCKET_VIEW_BOOL="true"
else
  DDP_BUCKET_VIEW_BOOL="false"
fi


if [[ ! -f "${CKPT}" ]]; then
  echo "[train_lpb_score_dsm] Missing policy checkpoint: ${CKPT}" >&2
  exit 1
fi

mkdir -p "${SAVE_DIR}"

if [[ "${USE_TMUX}" == "1" && -z "${TMUX:-}" ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[train_lpb_score_dsm] tmux is not installed; falling back to direct launch." >&2
  else
    SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
    INNER_CMD="$(
      printf '%q ' \
        env \
        USE_TMUX=0 \
        GPU="${GPU}" \
        SEED="${SEED}" \
        BASE_SAVE_DIR="${BASE_SAVE_DIR}" \
        TIMESTAMP="${TIMESTAMP}" \
        RUN_NAME="${RUN_NAME}" \
        SAVE_NAME="${SAVE_NAME}" \
        SAVE_DIR="${SAVE_DIR}" \
        CKPT="${CKPT}" \
        CUDA_PREFETCH="${CUDA_PREFETCH}" \
        DATA_IN_RAM="${DATA_IN_RAM}" \
        POSITIVE_RATIO="${POSITIVE_RATIO}" \
        HORIZON="${HORIZON}" \
        DSM_WINDOW_SIZE="${DSM_WINDOW_SIZE}" \
        TRAIN_ENCODER="${TRAIN_ENCODER}" \
        LORA_RANK="${LORA_RANK}" \
        LORA_ALPHA="${LORA_ALPHA}" \
        LORA_DROPOUT="${LORA_DROPOUT}" \
        DDP_FIND_UNUSED="${DDP_FIND_UNUSED}" \
        DDP_STATIC_GRAPH="${DDP_STATIC_GRAPH}" \
        DDP_BUCKET_VIEW="${DDP_BUCKET_VIEW}" \
        LOG_FILE="${LOG_FILE}" \
        PYTHON_BIN="${PYTHON_BIN}" \
        TORCHRUN_BIN="${TORCHRUN_BIN}" \
        MASTER_PORT="${MASTER_PORT:-29501}" \
        bash \
        "${SCRIPT_PATH}" \
        "$@"
    )"
    TRAIN_PANE_CMD="bash -lc '${INNER_CMD}; status=\$?; echo; echo \"[train_lpb_score_dsm] training exited with status=\${status}\"; exec bash'"
    GPU_PANE_CMD="bash -lc 'while true; do clear; date; echo; nvidia-smi; sleep 1; done'"
    SYS_PANE_CMD="bash -lc 'while true; do clear; date; echo; free -h; echo; ps -eo pid,ppid,%cpu,%mem,rss,cmd --sort=-rss | head -n 15; sleep 2; done'"
    LOG_PANE_CMD="bash -lc 'touch \"${LOG_FILE}\"; tail -n 80 -f \"${LOG_FILE}\"'"

    tmux new-session -d -s "${TMUX_SESSION_NAME}" "${TRAIN_PANE_CMD}"
    tmux split-window -h -t "${TMUX_SESSION_NAME}:0.0" "${GPU_PANE_CMD}"
    tmux split-window -v -t "${TMUX_SESSION_NAME}:0.1" "${SYS_PANE_CMD}"
    tmux split-window -v -t "${TMUX_SESSION_NAME}:0.0" "${LOG_PANE_CMD}"
    tmux select-layout -t "${TMUX_SESSION_NAME}:0" tiled >/dev/null
    tmux select-pane -t "${TMUX_SESSION_NAME}:0.0"
    echo "[train_lpb_score_dsm] launched tmux session=${TMUX_SESSION_NAME}"
    echo "[train_lpb_score_dsm] log_file=${LOG_FILE}"
    exec tmux attach-session -t "${TMUX_SESSION_NAME}"
  fi
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
IFS=',' read -r -a GPU_IDS <<< "${GPU}"
NUM_PROCS="${#GPU_IDS[@]}"
MASTER_PORT="${MASTER_PORT:-29501}"
LOCAL_NUM_WORKERS="${NUM_WORKERS}"
if [[ "${NUM_PROCS}" -gt 1 ]]; then
  LOCAL_NUM_WORKERS="$(( (NUM_WORKERS + NUM_PROCS - 1) / NUM_PROCS ))"
fi
LOCAL_TRAIN_BATCH_BUILD_WORKERS="${TRAIN_BATCH_BUILD_WORKERS:-2}"

echo "[train_lpb_score_dsm] policy.ckpt=${CKPT}"
echo "[train_lpb_score_dsm] save_dir=${SAVE_DIR}"
echo "[train_lpb_score_dsm] log_file=${LOG_FILE}"
echo "[train_lpb_score_dsm] batch_size=${BATCH_SIZE} encoder_batch_size=${ENCODER_BATCH_SIZE} num_workers=${LOCAL_NUM_WORKERS}/rank num_procs=${NUM_PROCS}"
echo "[train_lpb_score_dsm] window_size=${DSM_WINDOW_SIZE} num_pos=${NUM_POS} num_neg=${NUM_NEG} data_in_ram=${DATA_IN_RAM} cuda_prefetch=${CUDA_PREFETCH_BOOL}"
echo "[train_lpb_score_dsm] train_preload_ram_gb=${TRAIN_PRELOAD_RAM_GB} val_preload_ram_gb=${VAL_PRELOAD_RAM_GB}"
echo "[train_lpb_score_dsm] train_batch_build_workers=${LOCAL_TRAIN_BATCH_BUILD_WORKERS} batch_prefetch_depth=${BATCH_PREFETCH_DEPTH} batched_trajectory_cache_gb=${BATCHED_TRAJECTORY_CACHE_GB}"
echo "[train_lpb_score_dsm] preprocessed_cache_layout=${PREPROCESSED_CACHE_LAYOUT} upgrade_legacy_preprocessed_cache=${UPGRADE_LEGACY_PREPROCESSED_CACHE_BOOL}"
echo "[train_lpb_score_dsm] ddp_find_unused=${DDP_FIND_UNUSED_BOOL} ddp_static_graph=${DDP_STATIC_GRAPH_BOOL} ddp_bucket_view=${DDP_BUCKET_VIEW_BOOL}"

if [[ "${NUM_PROCS}" -le 1 ]]; then
  LAUNCHER=("${PYTHON_BIN}")
else
  LAUNCHER=(
    "${TORCHRUN_BIN}"
    --standalone
    --nproc_per_node="${NUM_PROCS}"
    --master_port="${MASTER_PORT}"
  )
fi

set -o pipefail
"${LAUNCHER[@]}" "${ROOT}/robosuite/discriminator/lpb_score/train.py" \
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
  dataset.window_size="${DSM_WINDOW_SIZE}" \
  data.splits.train.num_pos_traj="${NUM_POS}" \
  data.splits.train.num_neg_traj="${NUM_NEG}" \
  data.preprocessed_cache_layout="${PREPROCESSED_CACHE_LAYOUT}" \
  data.upgrade_legacy_preprocessed_cache="${UPGRADE_LEGACY_PREPROCESSED_CACHE_BOOL}" \
  training.batch_size="${BATCH_SIZE}" \
  training.num_workers="${LOCAL_NUM_WORKERS}" \
  training.data_in_ram="${DATA_IN_RAM}" \
  training.train_preload_ram_gb="${TRAIN_PRELOAD_RAM_GB}" \
  training.val_preload_ram_gb="${VAL_PRELOAD_RAM_GB}" \
  training.cuda_prefetch="${CUDA_PREFETCH_BOOL}" \
  training.batch_prefetch_depth="${BATCH_PREFETCH_DEPTH}" \
  training.train_batch_build_workers="${LOCAL_TRAIN_BATCH_BUILD_WORKERS}" \
  training.batched_trajectory_cache_gb="${BATCHED_TRAJECTORY_CACHE_GB}" \
  training.ddp_find_unused_parameters="${DDP_FIND_UNUSED_BOOL}" \
  training.ddp_static_graph="${DDP_STATIC_GRAPH_BOOL}" \
  training.ddp_gradient_as_bucket_view="${DDP_BUCKET_VIEW_BOOL}" \
  training.run_validation=false \
  training.epochs="${EPOCHS}" \
  training.lr="${LR}" \
  training.positive_ratio="${POSITIVE_RATIO}" \
  "$@" 2>&1 | tee -a "${LOG_FILE}"
