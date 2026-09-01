#!/usr/bin/env bash
# Visualize one round's published VAST and discriminator checkpoints.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
RUN_ROOT="outputs/dipole/PickPlaceCereal/multi-explore_20260804_233553"
ROUND_INDEX="001"
SEEDS=(1 2 3 4 5 6)
SPLIT="online"  # success_rollout, fail_rollout, or online

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"

cd "$ROOT_DIR"
exec "$PY" -m robosuite.pipeline.modules.visualization.vast.launch \
  "$RUN_ROOT" "$ROUND_INDEX" \
  --seeds "${SEEDS[@]}" \
  --split "$SPLIT"
