#!/usr/bin/env bash
# Build frozen DINOv3 latents, then pretrain RPT with two-GPU DDP.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/RPT/cache/dinov3_mean_v1}"
DINO_MODEL_PATH="${DINO_MODEL_PATH:-${REPO_ROOT}/data/pretrained/dinov3-vitb16-pretrain-lvd1689m_80M}"
TASKS="${TASKS:-NutAssemblyRound NutAssemblySquare PickPlaceBread PickPlaceCan PickPlaceCereal PickPlaceMilk Stack}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/dyn_disc/ablations/RPT/pretrain}"
RUN_NAME="${RUN_NAME:-rpt_robosuite}"
PRETRAIN_OUT_DIR="${PRETRAIN_OUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}-$(date +%Y%m%d_%H%M%S)}"

"${PYTHON_BIN}" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"'
[[ -e "${DINO_MODEL_PATH}" ]] || { echo "DINOv3 model not found: ${DINO_MODEL_PATH}" >&2; exit 1; }

TASK_ARGS=()
read -r -a TASK_ARGS <<< "${TASKS}"
INPUT_DIRS=()
for task in "${TASK_ARGS[@]}"; do
    input_dir="${DATA_ROOT}/${task}/success_rollout"
    [[ -d "${input_dir}" ]] || { echo "Training input not found: ${input_dir}" >&2; exit 1; }
    INPUT_DIRS+=("${input_dir}")
done

mkdir -p "${CACHE_ROOT}"
"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m robosuite.discriminator.dyn_disc.training.build_rpt_cache \
    --input "${INPUT_DIRS[@]}" \
    --cache-dir "${CACHE_ROOT}" \
    --model-path "${DINO_MODEL_PATH}" \
    --batch-size "${CACHE_BATCH_SIZE:-800}" \
    --num-workers "${CACHE_NUM_WORKERS:-2}" \
    --prefetch-factor "${CACHE_PREFETCH_FACTOR:-2}" \
    --io-chunk-frames "${CACHE_IO_CHUNK_FRAMES:-0}" \
    --max-trajectories-per-task "${MAX_TRAJECTORIES_PER_TASK:-100}"

export HYDRA_FULL_ERROR=1
"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m robosuite.discriminator.dyn_disc.training.train \
    --config-name train_rpt_robosuite \
    "cache.root=${CACHE_ROOT}" \
    "encoder.model_path=${DINO_MODEL_PATH}" \
    "run_name=${RUN_NAME}" \
    "hydra.run.dir=${PRETRAIN_OUT_DIR}" \
    "$@"
