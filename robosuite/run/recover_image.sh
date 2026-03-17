#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ROOT_DIR="/home/dodo/Documents/DAggar/robosuite"
INPUT_DIR="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert"
OUTPUT_DIR="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert_recover"
PYTHON_BIN="python"
JOBS=8

mkdir -p "${OUTPUT_DIR}"

find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.hdf5' -print0 | \
  xargs -0 -P "${JOBS}" -I {} bash -lc '
    in_file="$1"
    out_dir="$2"
    python_bin="$3"
    root_dir="$4"
    out_file="${out_dir}/$(basename "${in_file}")"

    "${python_bin}" "${root_dir}/robosuite/scripts/backfill_camera_from_states.py" \
      --input "${in_file}" \
      --all-cameras \
      --output "${out_file}"
  ' _ {} "${OUTPUT_DIR}" "${PYTHON_BIN}" "${ROOT_DIR}"
