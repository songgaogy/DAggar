#!/usr/bin/env bash
# Training entry for the dyn_disc dynamics model on robosuite HDF5 success data.
# Run from repo root:
#   TASK=PickPlaceCan bash robosuite/discriminator/dyn_disc/scripts/train_dyn_disc_robosuite_dynamics.sh
#
# Required overrides (Hydra):
#   env.train_data_path  -> built from data/<task>/success_rollout
# Other knobs override via Hydra CLI, e.g. training.epochs=20 frameskip=1

set -euo pipefail

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"


TASKS="NutAssemblyRound NutAssemblySquare PickPlaceBread PickPlaceCan PickPlaceCereal PickPlaceMilk Stack"
ACTION_STEPS=8
DATA_ROOT="${REPO_ROOT}/data"
TRAIN_SPLIT="success_rollout"
MAX_TRAJECTORIES_PER_TASK=100           # 0 -> no cap
TRAIN_ENCODER=0                         # 1 -> finetune the visual encoder
ENCODER_LR="${ENCODER_LR:-}"             # override training.encoder_lr; default = yaml (1.5e-4 = 30% of predictor_lr)
VIEW_NAMES="[agentview, robot0_eye_in_hand]"
VIEW_LOSS_NAMES="[agentview]"


TRAIN_DIRS=()
SKIPPED_TASKS=()
for t in ${TASKS}; do
    split_dir="${DATA_ROOT}/${t}/${TRAIN_SPLIT}"
    if compgen -G "${split_dir}/*.hdf5" >/dev/null; then
        TRAIN_DIRS+=("${split_dir}")
    else
        SKIPPED_TASKS+=("${t}")
    fi
done
if [[ "${#TRAIN_DIRS[@]}" -eq 0 ]]; then
    echo "[dyn_disc][train] ERROR: no ${TRAIN_SPLIT} HDF5 files found under DATA_ROOT=${DATA_ROOT} for TASKS=${TASKS}" >&2
    exit 1
fi

TRAIN_PATHS_OVERRIDE="["
for p in "${TRAIN_DIRS[@]}"; do
    TRAIN_PATHS_OVERRIDE+="${p},"
done
TRAIN_PATHS_OVERRIDE="${TRAIN_PATHS_OVERRIDE%,}]"

echo "[dyn_disc][train] using ${#TRAIN_DIRS[@]} ${TRAIN_SPLIT} dirs: ${TRAIN_DIRS[*]}"
if [[ "${#SKIPPED_TASKS[@]}" -gt 0 ]]; then
    echo "[dyn_disc][train] skipped tasks without ${TRAIN_SPLIT}: ${SKIPPED_TASKS[*]}"
fi

ACTION_DIM="${ACTION_DIM:-7}"
PROPRIO_DIM="${PROPRIO_DIM:-14}"
ORIGINAL_IMG_SIZE="${ORIGINAL_IMG_SIZE:-256}"
CROPPED_IMG_SIZE="${CROPPED_IMG_SIZE:-128}"
ACTION_EMB_DIM="${ACTION_EMB_DIM:-7}"
PROPRIO_EMB_DIM="${PROPRIO_EMB_DIM:-32}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FRAMESKIP="${FRAMESKIP:-1}"
RUN_NAME="${RUN_NAME:-train}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="./checkpoints/dyn_disc/dynamics/${RUN_NAME}-${TIMESTAMP}"


EXTRA_OVERRIDES=()
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    # Optional: slice full sim states when HDF5 stores flattened MuJoCo state.
    EXTRA_OVERRIDES+=("proprio_indices=${PROPRIO_INDICES}")
fi
if [[ -n "${MAX_TRAJECTORIES:-}" ]]; then
    EXTRA_OVERRIDES+=("max_trajectories=${MAX_TRAJECTORIES}")
fi
EXTRA_OVERRIDES+=("env.train_data_path=${TRAIN_PATHS_OVERRIDE}")
EXTRA_OVERRIDES+=("env.val_data_path=${TRAIN_PATHS_OVERRIDE}")
if [[ "${MAX_TRAJECTORIES_PER_TASK}" != "0" ]]; then
    EXTRA_OVERRIDES+=("max_trajectories=${MAX_TRAJECTORIES_PER_TASK}")
fi

EXTRA_OVERRIDES+=("run_name=${RUN_NAME}")
EXTRA_OVERRIDES+=("hydra.run.dir=${RUN_DIR}")
EXTRA_OVERRIDES+=('hydra.job.chdir=true')

if [[ "${TRAIN_ENCODER}" == "1" ]]; then
    # Finetune the visual encoder (do NOT disable use_pretrained_encoder).
    EXTRA_OVERRIDES+=("model.train_encoder=true")
    EXTRA_OVERRIDES+=("use_pretrained_encoder=true")
fi
if [[ -n "${ENCODER_LR}" ]]; then
    EXTRA_OVERRIDES+=("training.encoder_lr=${ENCODER_LR}")
fi

EXTRA_OVERRIDES+=("$@")

"${PYTHON_BIN}" -m robosuite.discriminator.dyn_disc.training.train \
    env=hdf5 \
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
