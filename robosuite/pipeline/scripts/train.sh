#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="/home/dodo/miniconda3/envs/dagger/bin/python"
cd "${ROOT_DIR}"
exec "${PYTHON}" -m robosuite.pipeline.train --config-name overall task=PickPlaceCereal "$@"
