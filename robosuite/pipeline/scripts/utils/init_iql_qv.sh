#!/usr/bin/env bash
# IQL Q/V offline warmup entry. Produces a per-task `iql_state.pt` that
# `train_dipole_rl.sh` can pick up via IQL_WARMUP_CKPT (which maps to the
# Hydra key `algorithm.q_learning.warmup_ckpt`).
#
# Required env vars:
#   INIT_CHECKPOINT   — flow-dagger / flow-multi base checkpoint (provides
#                        camera layout + policy action_dim).
# Optional env vars:
#   ENVIRONMENT       — task env name (default PickPlaceMilk).
#   LPB_CKPT          — overrides algorithm.discriminator.warm_start_ckpt
#                        (default: checkpoints/lpb_v2/bce_ckeckpoints/<ENVIRONMENT>/bce_head.pth).
#   LPB_META_JSON     — operational Youden threshold metadata (default: sibling
#                        meta.json with key bce_youden_threshold).
#   VALUE_STEPS       — V-only warmup loop length (default 20000).
#   FULL_STEPS        — full IQL update loop length (default 10000).
#   BATCH_SIZE        — minibatch size (default 128).
#   DEVICE            — learner device (default cuda:1).
#   OUTPUT_DIR        — per-task output dir (default outputs/DIPOLE_rl/iql_qv_cache/<ENVIRONMENT>).
#   NUM_TRAJECTORIES  — alias for NUM_TRAJECTORIES_EXPERT (legacy).
#   NUM_TRAJECTORIES_EXPERT / _SUCCESS / _FAIL — per-split HDF5 caps
#                        (unset = config defaults; <=0 = all).
#   WARMUP_DEMO_SPLITS — comma-separated HDF5 subdirs under data/<task>/
#                        (default: expert,success_rollout,fail_rollout).

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"

PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"

ENVIRONMENT="PickPlaceBread"

# Canonical per-task LPB v2 BCE layout (see pipeline/docs/IQL_DISCRIMINATOR_REWARD_DEBUG.md):
#   checkpoints/lpb_v2/bce_ckeckpoints/<TASK>/bce_head.pth
#   checkpoints/lpb_v2/bce_ckeckpoints/<TASK>/meta.json  → bce_youden_threshold
LPB_CKPT_DIR="${LPB_CKPT_DIR:-${ROOT_DIR}/checkpoints/lpb_v2/bce_ckeckpoints/${ENVIRONMENT}}"
LPB_CKPT="${LPB_CKPT:-${LPB_CKPT_DIR}/bce_head.pth}"
LPB_META_JSON="${LPB_META_JSON:-${LPB_CKPT_DIR}/meta.json}"
DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"

VALUE_STEPS="${VALUE_STEPS:-20000}"
FULL_STEPS="${FULL_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda:1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/DIPOLE_rl/iql_qv_cache/${ENVIRONMENT}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/iql_state.pt}"
INIT_CHECKPOINT="checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt"
mkdir -p "${OUTPUT_DIR}"

# IQL warmup is offline (HDF5 images + proprio); sparse r_env is replayed through
# the robosuite env per step (actual task success), not HDF5 split/last-frame labels.
# Env is built without offscreen rendering.
# MUJOCO_GL is unused unless you override warmup to enable rendering.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1


if [[ -z "${INIT_CHECKPOINT}" ]]; then
  echo "[ERROR] INIT_CHECKPOINT is required (flow-dagger / flow-multi base checkpoint)." >&2
  echo "        Example: INIT_CHECKPOINT=checkpoints/.../flow.pt bash $0" >&2
  exit 1
fi

HYDRA_OVERRIDES=(
  "env.environment=${ENVIRONMENT}"
  "data.task_name=${DEMO_TASK_NAME}"
  "runtime.init_checkpoint=${INIT_CHECKPOINT}"
  "algorithm.discriminator.warm_start_ckpt=${LPB_CKPT}"
  "+algorithm.discriminator.config.meta_json_path=${LPB_META_JSON}"
  "algorithm.q_learning.warmup_value_steps=${VALUE_STEPS}"
  "algorithm.q_learning.warmup_full_steps=${FULL_STEPS}"
  "algorithm.q_learning.config.device=${DEVICE}"
  "+warmup.output_path=${OUTPUT_FILE}"
  "+warmup.batch_size=${BATCH_SIZE}"
)

# set number of trajectories for each split
# NOTE: we do not need gt-fail label in iql training
NUM_TRAJECTORIES_EXPERT="${NUM_TRAJECTORIES_EXPERT:-${NUM_TRAJECTORIES:-}}"
if [[ -n "${NUM_TRAJECTORIES_EXPERT:-}" ]]; then
  HYDRA_OVERRIDES+=("warmup.num_trajectories.expert=${NUM_TRAJECTORIES_EXPERT}")
fi
if [[ -n "${NUM_TRAJECTORIES_SUCCESS:-}" ]]; then
  HYDRA_OVERRIDES+=("warmup.num_trajectories.success_rollout=${NUM_TRAJECTORIES_SUCCESS}")
fi
if [[ -n "${NUM_TRAJECTORIES_FAIL:-}" ]]; then
  HYDRA_OVERRIDES+=("warmup.num_trajectories.fail_rollout=${NUM_TRAJECTORIES_FAIL}")
fi

if [[ -n "${WARMUP_DEMO_SPLITS:-}" ]]; then
  IFS=',' read -r -a _warmup_splits <<< "${WARMUP_DEMO_SPLITS}"
  _warmup_splits_csv="$(IFS=','; echo "${_warmup_splits[*]}")"
  HYDRA_OVERRIDES+=("warmup.demo_splits=[${_warmup_splits_csv}]")
fi

if [[ ! -f "${LPB_CKPT}" ]]; then
  echo "[ERROR] LPB BCE checkpoint not found: ${LPB_CKPT}" >&2
  echo "        Set LPB_CKPT or ENVIRONMENT, or train/export under checkpoints/lpb_v2/bce_ckeckpoints/<TASK>/." >&2
  exit 1
fi
if [[ ! -f "${LPB_META_JSON}" ]]; then
  echo "[ERROR] LPB meta.json not found: ${LPB_META_JSON}" >&2
  echo "        Expected bce_youden_threshold next to bce_head.pth (not the threshold inside the .pth)." >&2
  exit 1
fi
LPB_TAU="$("${PY}" -c "import json, sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['bce_youden_threshold'])" "${LPB_META_JSON}")"

if [[ "${DEVICE}" == cuda* ]]; then
  if ! "${PY}" -c "import re, sys, torch; dev=sys.argv[1]; ok=torch.cuda.is_available(); m=re.fullmatch(r'cuda:(\\d+)', dev); ok = ok and (m is None or int(m.group(1)) < torch.cuda.device_count()); raise SystemExit(0 if ok else 1)" "${DEVICE}" 2>/dev/null; then
    echo "[ERROR] DEVICE=${DEVICE} is not available to torch." >&2
    echo "        Run: nvidia-smi   (fix driver/library mismatch; reboot after driver update)" >&2
    exit 1
  fi
fi

echo "[init_iql_qv] env=${ENVIRONMENT} device=${DEVICE}"
echo "[init_iql_qv] value_steps=${VALUE_STEPS} full_steps=${FULL_STEPS} batch=${BATCH_SIZE}"
echo "[init_iql_qv] output=${OUTPUT_FILE}"
echo "[init_iql_qv] init_checkpoint=${INIT_CHECKPOINT}"
echo "[init_iql_qv] lpb_ckpt=${LPB_CKPT}"
echo "[init_iql_qv] lpb_meta=${LPB_META_JSON} bce_youden_threshold=${LPB_TAU}"

"${PY}" -m robosuite.pipeline.algorithms.q_learning.warmup "${HYDRA_OVERRIDES[@]}"

echo "[init_iql_qv] done. Reuse with:"
echo "  IQL_WARMUP_CKPT=${OUTPUT_FILE} bash robosuite/pipeline/scripts/train_dipole_rl.sh"
