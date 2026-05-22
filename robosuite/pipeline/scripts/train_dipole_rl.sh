#!/usr/bin/env bash
# DIPOLE-RL training bash entry. Mirrors scripts/train_dipole.sh; adds the
# RL knobs.
#
# Required env vars:
#   INIT_CHECKPOINT  — path to flow-dagger / flow-multi base checkpoint
#                      (passed to runtime.init_checkpoint)
# Optional env vars:
#   LPB_CKPT         — overrides algorithm.discriminator.warm_start_ckpt
#   IQL_WARMUP_CKPT  — overrides algorithm.q_learning.warmup_ckpt
#   ALPHA, BETA      — overrides algorithm.advantage_g_provider.{alpha,beta}
#   G_MODE           — "advantage" | "bce_frozen"
#   INTERACTIVE / VIEWER_ENABLED / INTERVENTION_ENABLED  — same semantics
#                      as scripts/train_dipole.sh.

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"

if [[ "${INTERACTIVE:-true}" == "true" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "ERROR: INTERACTIVE=true but DISPLAY is empty" >&2
    exit 1
  fi
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-songgao-personal}"

PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"

if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
  echo "ERROR: INIT_CHECKPOINT is required (path to flow-dagger / flow-multi base checkpoint)." >&2
  exit 1
fi

HYDRA_ARGS=(
  "runtime.init_checkpoint=${INIT_CHECKPOINT}"
)
[[ -n "${LPB_CKPT:-}" ]]              && HYDRA_ARGS+=("algorithm.discriminator.warm_start_ckpt=${LPB_CKPT}")
[[ -n "${IQL_WARMUP_CKPT:-}" ]]       && HYDRA_ARGS+=("algorithm.q_learning.warmup_ckpt=${IQL_WARMUP_CKPT}")
[[ -n "${ALPHA:-}" ]]                 && HYDRA_ARGS+=("algorithm.advantage_g_provider.alpha=${ALPHA}")
[[ -n "${BETA:-}" ]]                  && HYDRA_ARGS+=("algorithm.advantage_g_provider.beta=${BETA}")
[[ -n "${G_MODE:-}" ]]                && HYDRA_ARGS+=("algorithm.dipole.g_mode=${G_MODE}")
[[ -n "${INTERACTIVE:-}" ]]           && HYDRA_ARGS+=("runtime.interactive=${INTERACTIVE}")
[[ -n "${VIEWER_ENABLED:-}" ]]        && HYDRA_ARGS+=("runtime.viewer_enabled=${VIEWER_ENABLED}")
[[ -n "${INTERVENTION_ENABLED:-}" ]]  && HYDRA_ARGS+=("intervention.enabled=${INTERVENTION_ENABLED}")

# Forward any additional CLI args (e.g. ``runtime.max_steps=500``).
exec "$PY" -m robosuite.pipeline.train_dipole_rl "${HYDRA_ARGS[@]}" "$@"
