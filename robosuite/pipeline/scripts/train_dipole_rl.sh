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

PY="${PY:-$HOME/miniconda3/envs/daggar/bin/python}"

# TODO(implementation): assemble Hydra override list from the optional env
# vars above and call:
#   "$PY" -m robosuite.pipeline.train_dipole_rl \
#       runtime.init_checkpoint=$INIT_CHECKPOINT \
#       algorithm.discriminator.warm_start_ckpt=$LPB_CKPT \
#       algorithm.q_learning.warmup_ckpt=$IQL_WARMUP_CKPT \
#       algorithm.advantage_g_provider.alpha=$ALPHA \
#       algorithm.advantage_g_provider.beta=$BETA \
#       algorithm.dipole.g_mode=$G_MODE \
#       runtime.interactive=$INTERACTIVE \
#       runtime.viewer_enabled=$VIEWER_ENABLED \
#       intervention.enabled=$INTERVENTION_ENABLED
echo "train_dipole_rl.sh skeleton — implementation pending (see docs/prompts/05_integrated_trainer.md)"
exit 1
