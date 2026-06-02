#!/usr/bin/env bash
# Visualize IQL Q/V on HDF5 rollout windows (see algorithms/q_learning/utils/vis_qv.py).
# Env vars mirror init_iql_qv.sh so LPB / IQL paths stay consistent.
#
# Optional env vars:
#   ENVIRONMENT       — task env name (default PickPlaceCereal, same as init_iql_qv.sh).
#   DEMO_TASK_NAME    — HDF5 subdir under data/ (default: ENVIRONMENT).
#   LPB_CKPT          — BCE head for r_disc + discriminator HUD (default: bce_ckeckpoints/<ENV>/bce_head.pth).
#   LPB_META_JSON     — Youden threshold metadata (default: sibling meta.json).
#   OUTPUT_DIR        — IQL warmup output dir (default: outputs/DIPOLE_rl/iql_qv_cache/<ENVIRONMENT>).
#   IQL_CKPT          — trained IQL state (default: ${OUTPUT_DIR}/iql_state.pt).
#   SPLIT             — HDF5 subdir split (default fail_rollout).
#   SEED              — demo selection seed (default 42).
#   DEVICE            — torch device (default cuda:1).
#   NO_DISC_VIZ       — set to 1 to skip discriminator/ HUD (LPB BCE benchmark path).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
cd "$ROOT_DIR"


ENVIRONMENT="PickPlaceCereal"
SEED=1
SPLIT="fail_rollout"    # success_rollout or fail_rollout
TARGET_PATH="iql_qv_cache-weight0_1"


DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"
LPB_CKPT_DIR="${LPB_CKPT_DIR:-${ROOT_DIR}/checkpoints/lpb_v2/bce_ckeckpoints/${ENVIRONMENT}}"
LPB_CKPT="${LPB_CKPT:-${LPB_CKPT_DIR}/bce_head.pth}"
LPB_META_JSON="${LPB_META_JSON:-${LPB_CKPT_DIR}/meta.json}"
TARGET_DIR="${ROOT_DIR}/outputs/DIPOLE_rl/${TARGET_PATH}/${ENVIRONMENT}"
IQL_CKPT="${TARGET_DIR}/iql_state.pt"
OUTPUT_DIR="${ROOT_DIR}/outputs/DIPOLE_rl/${TARGET_PATH}-vis"
DEVICE="${DEVICE:-cuda:0}"


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
