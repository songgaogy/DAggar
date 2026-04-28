#!/usr/bin/env bash
# Train LPB v2 dynamics on real-world Agilex cache.
# Build cache first:
#   ACTION_STOP=14 bash benchmark/real_world/scripts/build_cache.sh
# Then run:
#   bash robosuite/discriminator/lpb_v2/scripts/train_lpb_v2_agilex_dynamics.sh

set -euo pipefail

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

TASKS="${TASKS:-candy_in_plate duck_in_bowl Micky_in_box sausage_in_pot}"
CACHE_ROOT="${CACHE_ROOT:-data/.agilex_train_cache}"
TRAIN_SOURCES="${TRAIN_SOURCES:-success}"
MAX_TRAJECTORIES_PER_TASK=100
LOAD_ALL_INTO_RAM=1

TASKS_OVERRIDE="["
for t in ${TASKS}; do
    TASKS_OVERRIDE+="${t},"
done
TASKS_OVERRIDE="${TASKS_OVERRIDE%,}]"

SOURCES_OVERRIDE="["
for s in ${TRAIN_SOURCES}; do
    SOURCES_OVERRIDE+="${s},"
done
SOURCES_OVERRIDE="${SOURCES_OVERRIDE%,}]"

VIEW_NAMES="${VIEW_NAMES:-[cam_high]}"
ACTION_DIM="${ACTION_DIM:-7}"
PROPRIO_DIM="${PROPRIO_DIM:-7}"
ORIGINAL_IMG_SIZE="${ORIGINAL_IMG_SIZE:-224}"
CROPPED_IMG_SIZE="${CROPPED_IMG_SIZE:-224}"
ACTION_EMB_DIM="${ACTION_EMB_DIM:-7}"
PROPRIO_EMB_DIM="${PROPRIO_EMB_DIM:-32}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FRAMESKIP="${FRAMESKIP:-1}"
RUN_NAME="${RUN_NAME:-agilex_train}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="./checkpoints/lpb_v2/dynamics/${RUN_NAME}-${TIMESTAMP}"

EXTRA_OVERRIDES=()
EXTRA_OVERRIDES+=("env.cache_root=${CACHE_ROOT}")
EXTRA_OVERRIDES+=("env.tasks=${TASKS_OVERRIDE}")
EXTRA_OVERRIDES+=("env.train_sources=${SOURCES_OVERRIDE}")
EXTRA_OVERRIDES+=("env.load_all_into_ram=$([[ "${LOAD_ALL_INTO_RAM}" != "0" ]] && echo true || echo false)")
EXTRA_OVERRIDES+=("run_name=${RUN_NAME}")
EXTRA_OVERRIDES+=("hydra.run.dir=${RUN_DIR}")
EXTRA_OVERRIDES+=("hydra.job.chdir=true")

if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    EXTRA_OVERRIDES+=("proprio_indices=${PROPRIO_INDICES}")
fi
if [[ -n "${CAMERA_TO_VIEW:-}" ]]; then
    EXTRA_OVERRIDES+=("env.camera_to_view=${CAMERA_TO_VIEW}")
fi
if [[ "${MAX_TRAJECTORIES_PER_TASK}" != "0" ]]; then
    EXTRA_OVERRIDES+=("max_trajectories=${MAX_TRAJECTORIES_PER_TASK}")
fi

EXTRA_OVERRIDES+=("$@")

"${PYTHON_BIN}" -m robosuite.discriminator.lpb_v2.train \
    env=agilex \
    env.view_names="${VIEW_NAMES}" \
    env.action_dim="${ACTION_DIM}" \
    env.proprio_dim="${PROPRIO_DIM}" \
    env.original_img_size="${ORIGINAL_IMG_SIZE}" \
    env.cropped_img_size="${CROPPED_IMG_SIZE}" \
    env.action_emb_dim="${ACTION_EMB_DIM}" \
    env.proprio_emb_dim="${PROPRIO_EMB_DIM}" \
    training.epochs="${EPOCHS}" \
    training.batch_size="${BATCH_SIZE}" \
    frameskip="${FRAMESKIP}" \
    "${EXTRA_OVERRIDES[@]}"
