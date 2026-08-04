#!/usr/bin/env bash
# Evaluate and visualize one round's published discriminator checkpoint.
set -euo pipefail


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
RUN_ROOT="outputs/dipole/PickPlaceCereal/naive_trial_20260802_175238"
ROUND_INDEX="000"     # start from 000

cd "$ROOT_DIR"
exec "$PY" -m robosuite.pipeline.modules.evaluation.launch_discriminator \
  "$RUN_ROOT" "$ROUND_INDEX"
