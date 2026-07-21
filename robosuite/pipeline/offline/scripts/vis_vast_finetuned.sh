#!/usr/bin/env bash
# Visualize a finetuned offline VAST checkpoint with the same outputs as
# robosuite/pipeline/scripts/utils/vis_init_vast.sh.
#
# Experiment constants:
#   ENVIRONMENT, SEEDS, SPLIT, VAST_CKPT, and NNPU_CKPT are intentionally set
#   below. Edit them for a different offline run.
# Optional env vars:
#   DEMO_TASK_NAME       — task subdir under data/ (default: ENVIRONMENT).
#   OFFLINE_BUFFER       — vast_offline_transitions.pt override.
#   DEVICE               — torch CUDA device (default cuda:0).
#   NO_DISC_VIZ          — set to 1 to skip discriminator/ outputs.
#   VIDEO_FPS            — fps for rollout + discriminator HUD videos (default CONTROL_FREQ/20).
#   DISC_VIZ_IMAGE_SIZE  — square resolution for discriminator HUD MP4 (default 256).
#   DISC_VIZ_BORDER      — red-border thickness on predicted-failure frames (default 10).
#   GAE_LAMBDA           — λ for the GAE advantage overlaid on the TD advantage subplot (default 0.6).
#   MAX_WINDOWS          — optional cap for quick smoke tests.
#   VAST_SAMPLING_SEED   — optional diagnostic k-sampling seed override.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES=1

# -------------------------------------
ENVIRONMENT="${ENVIRONMENT:-PickPlaceCereal}"
SEEDS=(1 2 3 4 5 6)               # demo-selection seeds; one run per seed
SPLIT="fail"      # success or fail or online
VAST_CKPT="outputs/dipole-rl-offline_vast-disc_reward/PickPlaceCereal_20260719_212234/dipole/checkpoints/vast_state_finetuned.pt"
NNPU_CKPT="outputs/dipole-rl-offline_vast-disc_reward/PickPlaceCereal_20260719_212234/discriminator/checkpoints/pu_bce_head_finetuned.pth"
# -------------------------------------

DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"
OFFLINE_BUFFER_EXPLICIT=0
if [[ -n "${OFFLINE_BUFFER:-}" ]]; then
  OFFLINE_BUFFER_EXPLICIT=1
fi
OFFLINE_BUFFER="${OFFLINE_BUFFER:-${ROOT_DIR}/data/${DEMO_TASK_NAME}/offline_data-vast/vast_offline_transitions.pt}"
if [[ "${OFFLINE_BUFFER_EXPLICIT}" == "0" && ! -f "${OFFLINE_BUFFER}" ]]; then
  LEGACY_BUFFER="${ROOT_DIR}/data/${DEMO_TASK_NAME}/offline_data-vast/iql_offline_transitions.pt"
  if [[ -f "${LEGACY_BUFFER}" ]]; then
    echo "[WARN] Loading deprecated buffer ${LEGACY_BUFFER}." >&2
    OFFLINE_BUFFER="${LEGACY_BUFFER}"
  fi
fi
DEVICE="${DEVICE:-cuda:0}"
VIDEO_FPS="${VIDEO_FPS:-${CONTROL_FREQ:-20}}"
DISC_VIZ_BORDER="${DISC_VIZ_BORDER:-10}"
DISC_VIZ_IMAGE_SIZE="${DISC_VIZ_IMAGE_SIZE:-256}"
GAE_LAMBDA="${GAE_LAMBDA:-0.6}"
VAST_SAMPLING_SEED="${VAST_SAMPLING_SEED:-}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ -z "${VAST_CKPT}" || ! -f "${VAST_CKPT}" ]]; then
  echo "[ERROR] VAST_CKPT must point to vast_state_finetuned.pt: ${VAST_CKPT:-<unset>}" >&2
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
  --gae-lambda "${GAE_LAMBDA}"
)
if [[ -n "${VAST_SAMPLING_SEED}" ]]; then
  EXTRA_ARGS+=(--vast-sampling-seed "${VAST_SAMPLING_SEED}")
fi

echo "[vis_vast_finetuned] env=${ENVIRONMENT} task_data=${DEMO_TASK_NAME} split=${SPLIT} seeds=${SEEDS[*]}"
echo "[vis_vast_finetuned] checkpoint=${VAST_CKPT}"
echo "[vis_vast_finetuned] offline_buffer=${OFFLINE_BUFFER}"
echo "[vis_vast_finetuned] nnpu_ckpt=${NNPU_CKPT}"
echo "[vis_vast_finetuned] device=${DEVICE}"
echo "[vis_vast_finetuned] gae_lambda=${GAE_LAMBDA}"

for SEED in "${SEEDS[@]}"; do
  echo
  echo "[vis_vast_finetuned] === running seed=${SEED} ==="
  "${PY}" -m robosuite.pipeline.offline.utils.vis_vast_finetuned \
    --vast-ckpt "${VAST_CKPT}" \
    --task-data-name "${DEMO_TASK_NAME}" \
    --disc-ckpt "${NNPU_CKPT}" \
    --offline-buffer "${OFFLINE_BUFFER}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
done
