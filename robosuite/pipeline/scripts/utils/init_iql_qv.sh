#!/usr/bin/env bash
# IQL Q/V offline warmup entry. Produces a per-task `iql_state.pt` that
# `train_dipole_rl.sh` can pick up via IQL_WARMUP_CKPT (which maps to the
# Hydra key `algorithm.q_learning.warmup_ckpt`).
#
# Experiment constants:
#   ENVIRONMENT and INIT_CHECKPOINT are intentionally set below. Edit them for
#   a different task or base policy.
# Required env vars:
#   NNPU_CKPT         — task-calibrated pu_bce_head.pth.
# Optional env vars:
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

# -------------------------------------
ENVIRONMENT="PickPlaceMilk"
NAME="iql_qv_cache"
BRANCH_NAME="DIPOLE_rl"
# -------------------------------------

NNPU_CKPT="${NNPU_CKPT:-}"
DEMO_TASK_NAME="${DEMO_TASK_NAME:-${ENVIRONMENT}}"

VALUE_STEPS="${VALUE_STEPS:-20000}"
FULL_STEPS="${FULL_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DEVICE="${DEVICE:-cuda:1}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/${BRANCH_NAME}/${NAME}/${ENVIRONMENT}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/iql_state.pt}"
INIT_CHECKPOINT="checkpoints/multitask_6/flow_multi_ep0100.pt"
mkdir -p "${OUTPUT_DIR}"

# IQL warmup is offline (HDF5 images + proprio); sparse r_env is replayed through
# the robosuite env per step (actual task success), not HDF5 split/last-frame labels.
# Env is built without offscreen rendering.
# MUJOCO_GL is unused unless you override warmup to enable rendering.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1


if [[ -z "${INIT_CHECKPOINT}" ]]; then
  echo "[ERROR] Set INIT_CHECKPOINT in this script to a flow base checkpoint." >&2
  exit 1
fi

HYDRA_OVERRIDES=(
  "env.environment=${ENVIRONMENT}"
  "data.task_name=${DEMO_TASK_NAME}"
  "runtime.init_checkpoint=${INIT_CHECKPOINT}"
  "algorithm.discriminator.checkpoint=${NNPU_CKPT}"
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

if [[ -z "${NNPU_CKPT}" || ! -f "${NNPU_CKPT}" ]]; then
  echo "[ERROR] NNPU_CKPT must point to a task-calibrated pu_bce_head.pth: ${NNPU_CKPT:-<unset>}" >&2
  exit 1
fi

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
echo "[init_iql_qv] nnpu_ckpt=${NNPU_CKPT}"

"${PY}" -m robosuite.pipeline.algorithms.q_learning.warmup "${HYDRA_OVERRIDES[@]}"

echo "[init_iql_qv] done. Reuse with:"
echo "  IQL_WARMUP_CKPT=${OUTPUT_FILE} bash robosuite/pipeline/scripts/train_dipole_rl.sh"
