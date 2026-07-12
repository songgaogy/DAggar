#!/usr/bin/env bash
# Visualize IQL Q/V on the offline buffer exported by init_iql_qv.sh.
# Env vars mirror init_iql_qv.sh so nnPU / IQL paths stay consistent.
#
# Experiment constants:
#   ENVIRONMENT, SEED, SPLIT, TARGET_PATH, and BRANCH_NAME are intentionally
#   set below. Edit them for a different experiment.
# Optional env vars:
#   DEMO_TASK_NAME    — task subdir under data/ (default: ENVIRONMENT).
#   NNPU_CKPT         — nnPU head for r_disc + discriminator HUD.
#   OUTPUT_DIR        — IQL warmup output dir (default: outputs/DIPOLE_rl/iql_qv_cache/<ENVIRONMENT>).
#   IQL_CKPT          — trained IQL state (default: ${OUTPUT_DIR}/iql_state.pt).
#   OFFLINE_BUFFER    — saved warmup transitions (default: data/<task>/offline_data/iql_offline_transitions.pt).
#   SPLIT             — episode namespace filter (success_rollout | fail_rollout; BON only for success-like splits).
#   SEED              — offline episode selection seed (default 42).
#   DEVICE            — torch device (default cuda:1).
#   NO_DISC_VIZ       — set to 1 to skip the discriminator/ subdir (CSV+plot+HUD video).
#   VIS_BON           — 1/0: allow BON Q diagnostics on success-like splits (default 0).
#   VIDEO_FPS         — fps for rollout + discriminator HUD videos (default: CONTROL_FREQ/20).
#   DISC_VIZ_CAMERA   — camera to render for videos (default: agentview, else first policy cam).
#   DISC_VIZ_IMAGE_SIZE — square resolution for discriminator HUD MP4 (default: 256).
#   DISC_VIZ_BORDER   — red-border thickness on predicted-failure frames (default 10).
#   NO_FLIP_VERTICAL  — set to 1 to NOT flip rendered frames vertically.
#   MAX_WINDOWS             — optional cap for quick smoke tests.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES=0

# -------------------------------------
ENVIRONMENT="PickPlaceCereal"
SEEDS=(1 2 3)               # demo-selection seeds; one run per seed
SPLIT="success"      # success or fail
TARGET_PATH="offline_iql_qv-v2-01"
BRANCH_NAME="dipole_rl-iql"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth"
VIS_BON=0
# -------------------------------------

DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"
TARGET_DIR="${ROOT_DIR}/outputs/${BRANCH_NAME}/${TARGET_PATH}/${ENVIRONMENT}"
IQL_CKPT="${IQL_CKPT:-${TARGET_DIR}/iql_state.pt}"
OFFLINE_BUFFER="${OFFLINE_BUFFER:-${ROOT_DIR}/data/${DEMO_TASK_NAME}/offline_data-iql/iql_offline_transitions.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${BRANCH_NAME}/${TARGET_PATH}-vis}"
DEVICE="${DEVICE:-cuda}"
VIDEO_FPS="${VIDEO_FPS:-${CONTROL_FREQ:-20}}"
DISC_VIZ_BORDER="${DISC_VIZ_BORDER:-10}"
DISC_VIZ_IMAGE_SIZE="${DISC_VIZ_IMAGE_SIZE:-256}"


export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ -z "${NNPU_CKPT}" || ! -f "${NNPU_CKPT}" ]]; then
  echo "[ERROR] NNPU_CKPT must point to a task-calibrated pu_bce_head.pth: ${NNPU_CKPT:-<unset>}" >&2
  exit 1
fi

echo "[vis_iql_qv] env=${ENVIRONMENT} task_data=${DEMO_TASK_NAME} split=${SPLIT} seeds=${SEEDS[*]}"
echo "[vis_iql_qv] iql_ckpt=${IQL_CKPT}"
echo "[vis_iql_qv] offline_buffer=${OFFLINE_BUFFER}"
echo "[vis_iql_qv] nnpu_ckpt=${NNPU_CKPT}"
echo "[vis_iql_qv] device=${DEVICE}"

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

echo "[vis_iql_qv] vis_bon=${VIS_BON:-0}"

for SEED in "${SEEDS[@]}"; do
  echo "[vis_iql_qv] === running seed=${SEED} ==="
  "${PY}" -m robosuite.pipeline.algorithms.q_learning.utils.vis_qv \
    --iql-ckpt "${IQL_CKPT}" \
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
