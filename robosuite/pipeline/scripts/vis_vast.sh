#!/usr/bin/env bash
# Visualize one round's published VAST and discriminator checkpoints.
set -euo pipefail

RUN_ROOT=""
ROUND_INDEX=""
SEEDS=(1 2 3 4 5 6)
SPLIT="fail_rollout"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"

cd "$ROOT_DIR"
exec "$PY" -m robosuite.pipeline.modules.visualization.vast.launch \
  "$RUN_ROOT" "$ROUND_INDEX" \
  --seeds "${SEEDS[@]}" \
  --split "$SPLIT"
