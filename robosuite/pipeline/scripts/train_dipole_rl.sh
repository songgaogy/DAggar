#!/usr/bin/env bash
# DIPOLE-RL training bash entry. Mirrors scripts/train_dipole.sh; adds the
# RL knobs.
#
# Required env vars:
#   INIT_CHECKPOINT  — path to flow-dagger / flow-multi base checkpoint
#                      (passed to runtime.init_checkpoint)
# Optional env vars:
#   NNPU_CKPT        — task-calibrated pu_bce_head.pth
#   VAST_WARMUP_CKPT — overrides algorithm.vast.warmup_ckpt
#   IQL_WARMUP_CKPT  — deprecated read-only alias for VAST_WARMUP_CKPT
#   ALPHA, DISC_WEIGHT — overrides algorithm.adv.{alpha,disc_weight}
#   G_MODE           — "advantage" | "nnpu_frozen"
#   LOGGING_USE_TENSORBOARD / LOGGING_USE_WANDB          — logging backends
#   INTERACTIVE / VIEWER_ENABLED / INTERVENTION_ENABLED  — same semantics
#                      as scripts/train_dipole.sh.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

if [[ "${INTERACTIVE:-true}" == "true" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "ERROR: INTERACTIVE=true but DISPLAY is empty" >&2
    exit 1
  fi
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
LOGGING_USE_TENSORBOARD="${LOGGING_USE_TENSORBOARD:-true}"
LOGGING_USE_WANDB="${LOGGING_USE_WANDB:-false}"

if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
  echo "ERROR: INIT_CHECKPOINT is required (path to flow-dagger / flow-multi base checkpoint)." >&2
  exit 1
fi
if [[ -z "${NNPU_CKPT:-}" || ! -f "${NNPU_CKPT}" ]]; then
  echo "ERROR: NNPU_CKPT must point to a task-calibrated pu_bce_head.pth." >&2
  exit 1
fi

HYDRA_ARGS=(
  "runtime.init_checkpoint=${INIT_CHECKPOINT}"
  "algorithm.discriminator.checkpoint=${NNPU_CKPT}"
  "logging.use_tensorboard=${LOGGING_USE_TENSORBOARD}"
  "logging.use_wandb=${LOGGING_USE_WANDB}"
)
if [[ -n "${VAST_WARMUP_CKPT:-}" && -n "${IQL_WARMUP_CKPT:-}" ]]; then
  echo "ERROR: Set only VAST_WARMUP_CKPT; IQL_WARMUP_CKPT is a deprecated alias." >&2
  exit 1
fi
if [[ -n "${IQL_WARMUP_CKPT:-}" ]]; then
  echo "WARNING: IQL_WARMUP_CKPT is deprecated; use VAST_WARMUP_CKPT." >&2
  VAST_WARMUP_CKPT="${IQL_WARMUP_CKPT}"
fi
[[ -n "${NNPU_ENCODER_CKPT:-}" ]]     && HYDRA_ARGS+=("algorithm.discriminator.encoder_ckpt=${NNPU_ENCODER_CKPT}")
[[ -n "${VAST_WARMUP_CKPT:-}" ]]      && HYDRA_ARGS+=("algorithm.vast.warmup_ckpt=${VAST_WARMUP_CKPT}")
[[ -n "${ALPHA:-}" ]]                 && HYDRA_ARGS+=("algorithm.adv.alpha=${ALPHA}")
[[ -n "${DISC_WEIGHT:-}" ]]           && HYDRA_ARGS+=("algorithm.adv.disc_weight=${DISC_WEIGHT}")
[[ -n "${G_MODE:-}" ]]                && HYDRA_ARGS+=("algorithm.dipole.g_mode=${G_MODE}")
[[ -n "${INTERACTIVE:-}" ]]           && HYDRA_ARGS+=("runtime.interactive=${INTERACTIVE}")
[[ -n "${VIEWER_ENABLED:-}" ]]        && HYDRA_ARGS+=("runtime.viewer_enabled=${VIEWER_ENABLED}")
[[ -n "${INTERVENTION_ENABLED:-}" ]]  && HYDRA_ARGS+=("intervention.enabled=${INTERVENTION_ENABLED}")
HYDRA_ARGS+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

# Forward any additional CLI args (e.g. ``runtime.max_steps=500``).
exec "$PY" -m robosuite.pipeline.train_dipole_rl "${HYDRA_ARGS[@]}" "$@"
