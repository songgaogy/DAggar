#!/usr/bin/env bash
# Visualize IQL Q/V on HDF5 rollout windows (see algorithms/q_learning/utils/vis_qv.py).
# Env vars mirror init_iql_qv.sh so nnPU / IQL paths stay consistent.
#
# Experiment constants:
#   ENVIRONMENT, SEED, SPLIT, TARGET_PATH, and BRANCH_NAME are intentionally
#   set below. Edit them for a different experiment.
# Optional env vars:
#   DEMO_TASK_NAME    — HDF5 subdir under data/ (default: ENVIRONMENT).
#   NNPU_CKPT         — nnPU head for r_disc + discriminator HUD.
#   OUTPUT_DIR        — IQL warmup output dir (default: outputs/DIPOLE_rl/iql_qv_cache/<ENVIRONMENT>).
#   IQL_CKPT          — trained IQL state (default: ${OUTPUT_DIR}/iql_state.pt).
#   SPLIT             — HDF5 subdir split (success_rollout | fail_rollout; BON only for success_rollout).
#   SEED              — demo selection seed (default 42).
#   DEVICE            — torch device (default cuda:1).
#   NO_DISC_VIZ       — set to 1 to skip the discriminator/ subdir (CSV+plot+HUD video).
#   VIDEO_FPS         — fps for rollout + discriminator HUD videos (default: CONTROL_FREQ/20).
#   DISC_VIZ_CAMERA   — camera to render for videos (default: agentview, else first policy cam).
#   DISC_VIZ_BORDER   — red-border thickness on predicted-failure frames (default 10).
#   NO_FLIP_VERTICAL  — set to 1 to NOT flip rendered frames vertically.
#   Q_DIAG_NOISE_SIGMAS     — comma-separated full-chunk Gaussian sigmas (default 0.05,0.10,0.20).
#   Q_DIAG_NUM_RANDOM       — uniform random chunks per window (default 16).
#   Q_DIAG_SINGLE_DIM_SIGMA — Gaussian sigma for one-action-dim perturbation (default 0.20).
#   Q_DIAG_SINGLE_DIM_N     — number of action dims to perturb; 0 means all dims.
#   Q_DIAG_SEED             — candidate RNG seed (default: SEED).
#   MAX_WINDOWS             — optional cap for quick smoke tests.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES=0

# -------------------------------------
ENVIRONMENT="PickPlaceCereal"
SEED=6
SPLIT="success_rollout-val"    # success_rollout or fail_rollout
TARGET_PATH="offline_iql_qv-v2"
BRANCH_NAME="dipole_rl-debug"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth"
# -------------------------------------

DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"
TARGET_DIR="${ROOT_DIR}/outputs/${BRANCH_NAME}/${TARGET_PATH}/${ENVIRONMENT}"
IQL_CKPT="${TARGET_DIR}/iql_state.pt"
OUTPUT_DIR="${ROOT_DIR}/outputs/${BRANCH_NAME}/${TARGET_PATH}-vis"
DEVICE="${DEVICE:-cuda}"
# Q_DIAG_NOISE_SIGMAS="${Q_DIAG_NOISE_SIGMAS:-0.05,0.10,0.20}"
# Q_DIAG_NUM_RANDOM="${Q_DIAG_NUM_RANDOM:-16}"
# Q_DIAG_SINGLE_DIM_SIGMA="${Q_DIAG_SINGLE_DIM_SIGMA:-0.20}"
# Q_DIAG_SINGLE_DIM_N="${Q_DIAG_SINGLE_DIM_N:-0}"
# Q_DIAG_SEED="${Q_DIAG_SEED:-${SEED}}"
# Q_DIAG_ACTION_LOW="${Q_DIAG_ACTION_LOW:--1.0}"
# Q_DIAG_ACTION_HIGH="${Q_DIAG_ACTION_HIGH:-1.0}"
VIDEO_FPS="${VIDEO_FPS:-${CONTROL_FREQ:-20}}"
DISC_VIZ_BORDER="${DISC_VIZ_BORDER:-10}"


export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ -z "${NNPU_CKPT}" || ! -f "${NNPU_CKPT}" ]]; then
  echo "[ERROR] NNPU_CKPT must point to a task-calibrated pu_bce_head.pth: ${NNPU_CKPT:-<unset>}" >&2
  exit 1
fi

echo "[vis_iql_qv] env=${ENVIRONMENT} task_data=${DEMO_TASK_NAME} split=${SPLIT} seed=${SEED}"
echo "[vis_iql_qv] iql_ckpt=${IQL_CKPT}"
echo "[vis_iql_qv] nnpu_ckpt=${NNPU_CKPT}"
echo "[vis_iql_qv] device=${DEVICE}"

EXTRA_ARGS=()
if [[ -n "${MAX_WINDOWS:-}" ]]; then
  EXTRA_ARGS+=(--max-windows "${MAX_WINDOWS}")
fi
if [[ "${NO_DISC_VIZ:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-disc-viz)
fi
if [[ "${NO_FLIP_VERTICAL:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--no-flip-vertical)
fi
if [[ -n "${DISC_VIZ_CAMERA:-}" ]]; then
  EXTRA_ARGS+=(--disc-viz-camera "${DISC_VIZ_CAMERA}")
fi
EXTRA_ARGS+=(
  --video-fps "${VIDEO_FPS}"
  --disc-viz-border-thickness "${DISC_VIZ_BORDER}"
  # --q-candidate-noise-sigmas "${Q_DIAG_NOISE_SIGMAS}"
  # --q-candidate-random-n "${Q_DIAG_NUM_RANDOM}"
  # --q-candidate-single-dim-sigma "${Q_DIAG_SINGLE_DIM_SIGMA}"
  # --q-candidate-single-dim-n "${Q_DIAG_SINGLE_DIM_N}"
  # --q-candidate-seed "${Q_DIAG_SEED}"
  # --q-candidate-action-low "${Q_DIAG_ACTION_LOW}"
  # --q-candidate-action-high "${Q_DIAG_ACTION_HIGH}"
)

exec "${PY}" -m robosuite.pipeline.algorithms.q_learning.utils.vis_qv \
  --iql-ckpt "${IQL_CKPT}" \
  --output-root "${OUTPUT_DIR}" \
  --task-data-name "${DEMO_TASK_NAME}" \
  --disc-ckpt "${NNPU_CKPT}" \
  --split "${SPLIT}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
