#!/usr/bin/env bash
# Visualize IQL Q/V on HDF5 rollout windows (see algorithms/q_learning/utils/vis_qv.py).
# Env vars mirror init_iql_qv.sh so LPB / IQL paths stay consistent.
#
# Optional env vars:
#   ENVIRONMENT       — task env name (default PickPlaceCereal, same as init_iql_qv.sh).
#   DEMO_TASK_NAME    — HDF5 subdir under data/ (default: ENVIRONMENT).
#   LPB_CKPT          — BCE head for r_disc + discriminator HUD (default: bce_ckeckpoints/<ENV>/bce_head.pth).
#   LPB_META_JSON     — Youden threshold metadata (default: sibling meta.json).
#   OUTPUT_DIR        — visualization output dir.
#   IQL_CKPT          — trained IQL state (default: ${TARGET_DIR}/iql_state.pt).
#   SPLIT             — HDF5 subdir split (success_rollout | fail_rollout; BON only for success_rollout).
#   SEED              — demo selection seed (default 42).
#   DEVICE            — torch device (default cuda:1).
#   NO_DISC_VIZ       — set to 1 to skip discriminator/ HUD (LPB BCE benchmark path).
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

# -------------------------------------
ENVIRONMENT="PickPlaceCereal"
SEED=1
SPLIT="fail_rollout"
TARGET_PATH="iql_qv_cache-weight1_0"
ROOT_NAME="DIPOLE_rl-v1-aug-qv"
# -------------------------------------


DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"
LPB_CKPT_DIR="${LPB_CKPT_DIR:-${ROOT_DIR}/checkpoints/lpb_v2/bce_ckeckpoints/${ENVIRONMENT}}"
LPB_CKPT="${LPB_CKPT:-${LPB_CKPT_DIR}/bce_head.pth}"
LPB_META_JSON="${LPB_META_JSON:-${LPB_CKPT_DIR}/meta.json}"
TARGET_DIR="${TARGET_DIR:-${ROOT_DIR}/outputs/${ROOT_NAME}/${TARGET_PATH}/${ENVIRONMENT}}"
IQL_CKPT="${IQL_CKPT:-${TARGET_DIR}/iql_state.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${ROOT_NAME}/${TARGET_PATH}-vis}"
DEVICE="${DEVICE:-cuda:0}"
Q_DIAG_NOISE_SIGMAS="${Q_DIAG_NOISE_SIGMAS:-0.05,0.10,0.20}"
Q_DIAG_NUM_RANDOM="${Q_DIAG_NUM_RANDOM:-16}"
Q_DIAG_SINGLE_DIM_SIGMA="${Q_DIAG_SINGLE_DIM_SIGMA:-0.20}"
Q_DIAG_SINGLE_DIM_N="${Q_DIAG_SINGLE_DIM_N:-0}"
Q_DIAG_SEED="${Q_DIAG_SEED:-${SEED}}"
Q_DIAG_ACTION_LOW="${Q_DIAG_ACTION_LOW:--1.0}"
Q_DIAG_ACTION_HIGH="${Q_DIAG_ACTION_HIGH:-1.0}"


export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ ! -f "${IQL_CKPT}" ]]; then
  echo "[ERROR] IQL checkpoint not found: ${IQL_CKPT}" >&2
  echo "        Run init first, e.g.:" >&2
  echo "          ENVIRONMENT=${ENVIRONMENT} bash robosuite/pipeline/scripts/utils/init_iql_qv.sh" >&2
  exit 1
fi
if [[ ! -f "${LPB_CKPT}" ]]; then
  echo "[ERROR] LPB BCE checkpoint not found: ${LPB_CKPT}" >&2
  echo "        Set LPB_CKPT or ENVIRONMENT (same layout as init_iql_qv.sh)." >&2
  exit 1
fi
if [[ ! -f "${LPB_META_JSON}" ]]; then
  echo "[ERROR] LPB meta.json not found: ${LPB_META_JSON}" >&2
  echo "        Expected bce_youden_threshold next to bce_head.pth." >&2
  exit 1
fi
LPB_TAU="$("${PY}" -c "import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['bce_youden_threshold'])" "${LPB_META_JSON}")"


echo "[vis_iql_qv] env=${ENVIRONMENT} task_data=${DEMO_TASK_NAME} split=${SPLIT} seed=${SEED}"
echo "[vis_iql_qv] iql_ckpt=${IQL_CKPT}"
echo "[vis_iql_qv] lpb_ckpt=${LPB_CKPT}"
echo "[vis_iql_qv] lpb_meta=${LPB_META_JSON} bce_youden_threshold=${LPB_TAU}"
echo "[vis_iql_qv] device=${DEVICE}"
echo "[vis_iql_qv] q_diag_noise_sigmas=${Q_DIAG_NOISE_SIGMAS} q_diag_random_n=${Q_DIAG_NUM_RANDOM} single_dim_sigma=${Q_DIAG_SINGLE_DIM_SIGMA} single_dim_n=${Q_DIAG_SINGLE_DIM_N} q_diag_seed=${Q_DIAG_SEED}"

EXTRA_ARGS=()
if [[ -n "${MAX_WINDOWS:-}" ]]; then
  EXTRA_ARGS+=(--max-windows "${MAX_WINDOWS}")
fi
EXTRA_ARGS+=(
  --q-candidate-noise-sigmas "${Q_DIAG_NOISE_SIGMAS}"
  --q-candidate-random-n "${Q_DIAG_NUM_RANDOM}"
  --q-candidate-single-dim-sigma "${Q_DIAG_SINGLE_DIM_SIGMA}"
  --q-candidate-single-dim-n "${Q_DIAG_SINGLE_DIM_N}"
  --q-candidate-seed "${Q_DIAG_SEED}"
  --q-candidate-action-low "${Q_DIAG_ACTION_LOW}"
  --q-candidate-action-high "${Q_DIAG_ACTION_HIGH}"
)

exec "${PY}" -m robosuite.pipeline.algorithms.q_learning.utils.vis_qv \
  --iql-ckpt "${IQL_CKPT}" \
  --output-root "${OUTPUT_DIR}" \
  --task-data-name "${DEMO_TASK_NAME}" \
  --disc-ckpt "${LPB_CKPT}" \
  --split "${SPLIT}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
