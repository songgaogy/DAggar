#!/usr/bin/env bash
# Shared Hydra CLI overrides: no multirun dirs, no .hydra snapshots, no Hydra log noise.
#
# Usage (after ROOT_DIR is set and repo root is the working directory):
#   source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"
#   HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")
#
# Opt back into default Hydra outputs with HYDRA_ENABLE_OUTPUTS=1 before sourcing.

if [[ "${HYDRA_ENABLE_OUTPUTS:-0}" == "1" ]]; then
  HYDRA_DISABLE_LOG_OVERRIDES=()
else
  HYDRA_DISABLE_LOG_OVERRIDES=(
    hydra.run.dir=.
    hydra.sweep.dir=.
    hydra.job.chdir=false
    hydra.output_subdir=null
    hydra/hydra_logging=none
    hydra/job_logging=none
  )
fi
