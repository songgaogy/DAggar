#!/usr/bin/env bash
# Visualize VAST G/V stitching on the offline buffer exported by init_vast.sh.
# Env vars mirror init_vast.sh so nnPU / VAST paths stay consistent.
#
# Experiment constants:
#   ENVIRONMENT, SEED, SPLIT, TARGET_PATH, and BRANCH_NAME are intentionally
#   set below. Edit them for a different experiment.
# Optional env vars:
#   DEMO_TASK_NAME    — task subdir under data/ (default: ENVIRONMENT).
#   NNPU_CKPT         — nnPU head for r_disc + discriminator HUD.
#   OUTPUT_DIR        — viz output root (default: outputs/<BRANCH>/<TARGET>/<ENVIRONMENT>/vis).
#   VAST_CKPT         — trained VAST state (default: ${TARGET_DIR}/vast_state.pt).
#   IQL_CKPT          — deprecated read-only alias for VAST_CKPT.
#   OFFLINE_BUFFER    — saved warmup transitions (default: data/<task>/offline_data-vast/vast_offline_transitions.pt).
#   SPLIT             — episode namespace filter (success_rollout | fail_rollout).
#   SEED              — offline episode selection seed (default 42).
#   DEVICE            — torch device (default cuda:1).
#   NO_DISC_VIZ       — set to 1 to skip the discriminator/ subdir (CSV+plot+HUD video).
#   VIDEO_FPS         — fps for rollout + discriminator HUD videos (default: CONTROL_FREQ/20).
#   DISC_VIZ_CAMERA   — camera to render for videos (default: agentview, else first policy cam).
#   DISC_VIZ_IMAGE_SIZE — square resolution for discriminator HUD MP4 (default: 256).
#   DISC_VIZ_BORDER   — red-border thickness on predicted-failure frames (default 10).
#   GAE_LAMBDA        — λ for the GAE advantage overlaid on the TD advantage subplot (default 0.6).
#   VAST_SAMPLING_SEED — optional diagnostic k-sampling seed override.
#   NO_FLIP_VERTICAL  — set to 1 to NOT flip rendered frames vertically.
#   MAX_WINDOWS             — optional cap for quick smoke tests.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES=1

BRANCH_NAME="dipole_rl-vast"

# -------------------------------------
ENVIRONMENT="PickPlaceCereal"
SEEDS=(1 2 3 4 5 6)               # demo-selection seeds; one run per seed
SPLIT="fail"      # success or fail
TARGET_PATH="tau0p7_en5"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth"
# -------------------------------------

DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"
TARGET_DIR="${ROOT_DIR}/outputs/${BRANCH_NAME}/${TARGET_PATH}/${ENVIRONMENT}"
VAST_CKPT_EXPLICIT=0
if [[ -n "${VAST_CKPT:-}" || -n "${IQL_CKPT:-}" ]]; then
  VAST_CKPT_EXPLICIT=1
fi
if [[ -n "${VAST_CKPT:-}" && -n "${IQL_CKPT:-}" ]]; then
  echo "[ERROR] Set only VAST_CKPT; IQL_CKPT is a deprecated alias." >&2
  exit 1
fi
if [[ -n "${IQL_CKPT:-}" ]]; then
  echo "[WARN] IQL_CKPT is deprecated; use VAST_CKPT." >&2
  VAST_CKPT="${IQL_CKPT}"
fi
VAST_CKPT="${VAST_CKPT:-${TARGET_DIR}/vast_state.pt}"
if [[ "${VAST_CKPT_EXPLICIT}" == "0" && ! -f "${VAST_CKPT}" ]]; then
  LEGACY_CKPT="${TARGET_DIR}/iql_state.pt"
  if [[ -f "${LEGACY_CKPT}" ]]; then
    echo "[WARN] Loading deprecated checkpoint ${LEGACY_CKPT}; use vast_state.pt." >&2
    VAST_CKPT="${LEGACY_CKPT}"
  fi
fi
OFFLINE_BUFFER_EXPLICIT=0
if [[ -n "${OFFLINE_BUFFER:-}" ]]; then
  OFFLINE_BUFFER_EXPLICIT=1
fi
OFFLINE_BUFFER="${OFFLINE_BUFFER:-${ROOT_DIR}/data/${DEMO_TASK_NAME}/offline_data-vast/vast_offline_transitions.pt}"
if [[ "${OFFLINE_BUFFER_EXPLICIT}" == "0" && ! -f "${OFFLINE_BUFFER}" ]]; then
  LEGACY_BUFFER="${ROOT_DIR}/data/${DEMO_TASK_NAME}/offline_data-vast/iql_offline_transitions.pt"
  if [[ -f "${LEGACY_BUFFER}" ]]; then
    echo "[WARN] Loading deprecated buffer ${LEGACY_BUFFER}; use vast_offline_transitions.pt." >&2
    OFFLINE_BUFFER="${LEGACY_BUFFER}"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-${TARGET_DIR}/vis}"
DEVICE="${DEVICE:-cuda}"
VIDEO_FPS="${VIDEO_FPS:-${CONTROL_FREQ:-20}}"
DISC_VIZ_BORDER="${DISC_VIZ_BORDER:-10}"
DISC_VIZ_IMAGE_SIZE="${DISC_VIZ_IMAGE_SIZE:-256}"
GAE_LAMBDA="${GAE_LAMBDA:-0.6}"   # λ for the GAE advantage overlaid on the TD advantage subplot
VAST_SAMPLING_SEED="${VAST_SAMPLING_SEED:-}"


export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ -z "${NNPU_CKPT}" || ! -f "${NNPU_CKPT}" ]]; then
  echo "[ERROR] NNPU_CKPT must point to a task-calibrated pu_bce_head.pth: ${NNPU_CKPT:-<unset>}" >&2
  exit 1
fi

echo "[vis_vast] env=${ENVIRONMENT} task_data=${DEMO_TASK_NAME} split=${SPLIT} seeds=${SEEDS[*]}"
echo "[vis_vast] vast_ckpt=${VAST_CKPT}"
echo "[vis_vast] offline_buffer=${OFFLINE_BUFFER}"
echo "[vis_vast] output_dir=${OUTPUT_DIR}"
echo "[vis_vast] nnpu_ckpt=${NNPU_CKPT}"
echo "[vis_vast] device=${DEVICE}"
echo "[vis_vast] gae_lambda=${GAE_LAMBDA}"

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
for SEED in "${SEEDS[@]}"; do
  echo
  echo "[vis_vast] === running seed=${SEED} ==="
  "${PY}" -m robosuite.pipeline.algorithms.vast.utils.vis_vast \
    --vast-ckpt "${VAST_CKPT}" \
    --output-root "${OUTPUT_DIR}" \
    --task-data-name "${DEMO_TASK_NAME}" \
    --disc-ckpt "${NNPU_CKPT}" \
    --offline-buffer "${OFFLINE_BUFFER}" \
    --split "${SPLIT}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
done
