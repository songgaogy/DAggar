#!/usr/bin/env bash
# Evaluate and visualize one round's published discriminator checkpoint.
set -euo pipefail


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="/home/dodo/miniconda3/envs/dagger/bin/python"
RUN_ROOT=""
ROUND_INDEX=""

cd "$ROOT_DIR"
exec "$PY" -m robosuite.pipeline.modules.evaluation.launch_discriminator \
  "$RUN_ROOT" "$ROUND_INDEX"
