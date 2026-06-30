#!/usr/bin/env bash
# Open-loop action visualization for a pretrained flow-dagger policy on expert
# pretrain HDF5 data. Runs non-overlapping chunks: obs[0] -> actions[0:8],
# obs[8] -> actions[8:16], etc., and compares policy actions against GT.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "${ROOT_DIR}"

PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# -------------------------------------
TASK="${TASK:-PickPlaceCereal}"
CKPT="${CKPT:-checkpoints/multitask_6/flow_multi_ep0100.pt}"
DATA_DIR="${DATA_DIR:-data/PickPlaceCereal/pretrain_data-20260615_174814/expert_pretrain_data.hdf5}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/PickPlaceCereal/pretrain_data-20260615_174814}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
DEVICE="${DEVICE:-cuda:0}"
DETERMINISTIC="${DETERMINISTIC:-false}"
SEED="${SEED:-42}"
MAX_DEMOS="${MAX_DEMOS:-0}"
MAX_FRAMES="${MAX_FRAMES:-0}"
MAX_PLOT_POINTS="${MAX_PLOT_POINTS:-10000}"
# -------------------------------------

timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
ckpt_stem="$(basename "${CKPT}")"
ckpt_stem="${ckpt_stem%.*}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/openloop_vis/${ckpt_stem}__chunk_${CHUNK_SIZE}__${timestamp}}"

ARGS=(
  --checkpoint "${CKPT}"
  --data "${DATA_DIR}"
  --task-name "${TASK}"
  --env-name "${TASK}"
  --output-dir "${OUTPUT_DIR}"
  --chunk-size "${CHUNK_SIZE}"
  --max-demos "${MAX_DEMOS}"
  --max-frames "${MAX_FRAMES}"
  --max-plot-points "${MAX_PLOT_POINTS}"
  --device "${DEVICE}"
  --seed "${SEED}"
)

if [[ "${DETERMINISTIC}" == "true" ]]; then
  ARGS+=(--deterministic)
fi

echo "[openloop_vis] task=${TASK} ckpt=${CKPT}"
echo "[openloop_vis] data=${DATA_DIR}"
echo "[openloop_vis] chunk_size=${CHUNK_SIZE} max_demos=${MAX_DEMOS} max_chunk_starts=${MAX_FRAMES}"
echo "[openloop_vis] output_dir=${OUTPUT_DIR}"

"${PY}" -m robosuite.pipeline.dev.openloop_vis "${ARGS[@]}"
