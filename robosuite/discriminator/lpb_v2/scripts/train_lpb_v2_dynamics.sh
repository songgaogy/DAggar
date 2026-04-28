#!/usr/bin/env bash
# Training entry for the LPB v2 dynamics model on user's data.
# Run from repo root:
#   TASK=PickPlaceCan bash robosuite/discriminator/lpb_v2/scripts/train_lpb_v2_dynamics.sh
#
# Required overrides (Hydra):
#   env.train_data_path  -> built from TASK + rollout roots
# Other knobs override via Hydra CLI, e.g. training.epochs=20 frameskip=4

set -euo pipefail

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# By default, pool data from the same 6 tasks as LPB.
# You can set TASK=... to run a single task instead.
TASKS_DEFAULT="PandaLift PandaPickPlaceCan PandaStack PickPlaceBread PickPlaceCereal PickPlaceMilk"
if [[ -n "${TASK:-}" ]]; then
    TASKS="${TASK}"
else
    TASKS="${TASKS:-${TASKS_DEFAULT}}"
fi
# Training data comes from the preprocessed cache `.npz` files.
CACHE_ROOT="${CACHE_ROOT:-data/.lpb_score_preprocessed_cache}"
METADATA_CACHE_ROOT="${METADATA_CACHE_ROOT:-data/.lpb_score_cache}"
MAX_TRAJECTORIES_PER_TASK="${MAX_TRAJECTORIES_PER_TASK:-0}"  # 0 -> no cap
LOAD_ALL_INTO_RAM=1
NUM_EXPERT="${NUM_EXPERT:-0}"      # per task; -1 -> no cap, 0 -> use none
NUM_SUCCESS="${NUM_SUCCESS:-100}"  # per task; -1 -> no cap, 0 -> use none
NUM_FAIL="${NUM_FAIL:-0}"          # per task; -1 -> no cap, 0 -> use none
TRAIN_ENCODER="${TRAIN_ENCODER:-1}"  # 1 -> finetune ResNet encoder

# Build Hydra list literal for tasks: [a,b,c]
TASKS_OVERRIDE="["
for t in ${TASKS}; do
    TASKS_OVERRIDE+="${t},"
done
TASKS_OVERRIDE="${TASKS_OVERRIDE%,}]"

VIEW_NAMES="${VIEW_NAMES:-[agentview]}"
ACTION_DIM="${ACTION_DIM:-7}"
PROPRIO_DIM="${PROPRIO_DIM:-14}"
ORIGINAL_IMG_SIZE="${ORIGINAL_IMG_SIZE:-128}"
CROPPED_IMG_SIZE="${CROPPED_IMG_SIZE:-128}"
ACTION_EMB_DIM="${ACTION_EMB_DIM:-7}"
PROPRIO_EMB_DIM="${PROPRIO_EMB_DIM:-32}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FRAMESKIP="${FRAMESKIP:-1}"
RUN_NAME="${RUN_NAME:-train}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="./checkpoints/lpb_v2/dynamics/${RUN_NAME}-${TIMESTAMP}"


EXTRA_OVERRIDES=()
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    # Hydra list override form: proprio_indices=[0,1,2,3]
    EXTRA_OVERRIDES+=("proprio_indices=${PROPRIO_INDICES}")
fi
if [[ -n "${MAX_TRAJECTORIES:-}" ]]; then
    EXTRA_OVERRIDES+=("max_trajectories=${MAX_TRAJECTORIES}")
fi

if [[ -n "${CACHE_ROOT}" ]]; then
    EXTRA_OVERRIDES+=("env.cache_root=${CACHE_ROOT}")
fi
if [[ -n "${METADATA_CACHE_ROOT}" ]]; then
    EXTRA_OVERRIDES+=("env.metadata_cache_root=${METADATA_CACHE_ROOT}")
fi
EXTRA_OVERRIDES+=("env.num_expert=${NUM_EXPERT}")
EXTRA_OVERRIDES+=("env.num_success=${NUM_SUCCESS}")
EXTRA_OVERRIDES+=("env.num_fail=${NUM_FAIL}")
if [[ "${LOAD_ALL_INTO_RAM}" != "0" ]]; then
    EXTRA_OVERRIDES+=("env.load_all_into_ram=true")
else
    EXTRA_OVERRIDES+=("env.load_all_into_ram=false")
fi
EXTRA_OVERRIDES+=("env.tasks=${TASKS_OVERRIDE}")
if [[ "${MAX_TRAJECTORIES_PER_TASK}" != "0" ]]; then
    EXTRA_OVERRIDES+=("max_trajectories=${MAX_TRAJECTORIES_PER_TASK}")
fi

EXTRA_OVERRIDES+=("run_name=${RUN_NAME}")
EXTRA_OVERRIDES+=("hydra.run.dir=${RUN_DIR}")
EXTRA_OVERRIDES+=('hydra.job.chdir=true')

if [[ "${TRAIN_ENCODER}" == "1" ]]; then
    # Enable finetuning of the ResNet encoder.
    EXTRA_OVERRIDES+=("model.train_encoder=true")
    EXTRA_OVERRIDES+=("use_pretrained_encoder=false")
fi

EXTRA_OVERRIDES+=("$@")

"${PYTHON_BIN}" -m robosuite.discriminator.lpb_v2.train \
    env=preprocessed \
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
