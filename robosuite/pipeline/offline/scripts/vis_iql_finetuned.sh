#!/usr/bin/env bash
# Visualize a finetuned offline-IQL checkpoint with the same outputs as
# robosuite/pipeline/scripts/utils/vis_iql_qv.sh.
#
# Experiment constants:
#   ENVIRONMENT, SEEDS, SPLIT, CHECKPOINT, and NNPU_CKPT are intentionally set
#   below. Edit them for a different offline run.
# Optional env vars:
#   DEMO_TASK_NAME       — task subdir under data/ (default: ENVIRONMENT).
#   OFFLINE_BUFFER       — iql_offline_transitions.pt override (default: run_info.json).
#   DEVICE               — torch CUDA device (default cuda:0).
#   NO_DISC_VIZ          — set to 1 to skip discriminator/ outputs.
#   VIS_BON              — 1/0: allow BON Q diagnostics on success-like splits (default 0).
#   VIDEO_FPS            — fps for rollout + discriminator HUD videos (default CONTROL_FREQ/20).
#   DISC_VIZ_IMAGE_SIZE  — square resolution for discriminator HUD MP4 (default 256).
#   DISC_VIZ_BORDER      — red-border thickness on predicted-failure frames (default 10).
#   MAX_WINDOWS          — optional cap for quick smoke tests.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES=1

# -------------------------------------
ENVIRONMENT="PickPlaceCereal"
SEEDS=(1 2 3 4 5 6 7 8)               # demo-selection seeds; one run per seed
SPLIT="fail"      # success or fail or online
CHECKPOINT="outputs/dipole-rl-offline/PickPlaceCereal_20260711_214351_beta4_wo-success/checkpoints/iql_state_finetuned.pt"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth"
VIS_BON=0
# -------------------------------------

DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"
DEVICE="${DEVICE:-cuda:0}"
VIDEO_FPS="${VIDEO_FPS:-${CONTROL_FREQ:-20}}"
DISC_VIZ_BORDER="${DISC_VIZ_BORDER:-10}"
DISC_VIZ_IMAGE_SIZE="${DISC_VIZ_IMAGE_SIZE:-256}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] CHECKPOINT must point to iql_state_finetuned.pt: ${CHECKPOINT:-<unset>}" >&2
  exit 1
fi

if [[ -z "${NNPU_CKPT}" || ! -f "${NNPU_CKPT}" ]]; then
  echo "[ERROR] NNPU_CKPT must point to a task-calibrated pu_bce_head.pth: ${NNPU_CKPT:-<unset>}" >&2
  exit 1
fi

EXTRA_ARGS=()
EXTRA_ARGS+=(
  --video-fps "${VIDEO_FPS}"
  --disc-viz-image-size "${DISC_VIZ_IMAGE_SIZE}"
  --disc-viz-border-thickness "${DISC_VIZ_BORDER}"
)
# VIS_BON=1 allows BON; still only runs on success-like splits (vis_qv).
case "${VIS_BON:-0}" in
  1|true|True|TRUE) ;;
  *) EXTRA_ARGS+=(--no-bon) ;;
esac

echo "[vis_iql_finetuned] env=${ENVIRONMENT} task_data=${DEMO_TASK_NAME} split=${SPLIT} seeds=${SEEDS[*]}"
echo "[vis_iql_finetuned] checkpoint=${CHECKPOINT}"
echo "[vis_iql_finetuned] nnpu_ckpt=${NNPU_CKPT}"
echo "[vis_iql_finetuned] device=${DEVICE}"
echo "[vis_iql_finetuned] vis_bon=${VIS_BON:-0}"

for SEED in "${SEEDS[@]}"; do
  echo "[vis_iql_finetuned] === running seed=${SEED} ==="
  "${PY}" -m robosuite.pipeline.offline.utils.vis_iql_finetuned \
    --checkpoint "${CHECKPOINT}" \
    --task-data-name "${DEMO_TASK_NAME}" \
    --disc-ckpt "${NNPU_CKPT}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
done
