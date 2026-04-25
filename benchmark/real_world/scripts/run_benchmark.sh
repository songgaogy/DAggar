#!/usr/bin/env bash
# Real-world (agilex) failure-detection benchmark entry point.
# Run from repo root:
#   bash benchmark/real_world/scripts/run_benchmark.sh
# Override via env vars, e.g.:
#   TASKS="candy_in_plate" SCORE_MODE=linear-time MAX_SUCCESS_PER_TASK=20 \
#     bash benchmark/real_world/scripts/run_benchmark.sh
#
# Defaults wire to the agilex dataset that lives in this repo at:
#   ${REPO_ROOT}/data/agilex/failure_annotations/out_by_task/<task>/out.hdf5
#   ${REPO_ROOT}/data/agilex/<task>/success_rollout/episode_*.hdf5

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/agilex/failure_annotations/out_by_task}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/agilex}"
TASKS="${TASKS:-candy_in_plate duck_in_bowl Micky_in_box sausage_in_pot}"

# Per-task trajectory caps. Empty / non-positive int = use all.
MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-25}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-50}"

RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/real_world_benchmark/${RUN_NAME}}"
SAVE_JSON="${SAVE_JSON:-${OUT_DIR}/benchmark.json}"
mkdir -p "${OUT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# Right-arm slicing (action 6-D, proprio 7-D).
ACTION_START="${ACTION_START:-7}"
ACTION_STOP="${ACTION_STOP:-13}"
PROPRIO_FIELD="${PROPRIO_FIELD:-qpos}"
PROPRIO_START="${PROPRIO_START:-7}"
PROPRIO_STOP="${PROPRIO_STOP:-14}"

CAMERA_NAME="${CAMERA_NAME:-cam_high}"
SCORE_MODE="${SCORE_MODE:-random}"
SEED="${SEED:-0}"

EXTRA_ARGS=()
if [[ -n "${MAX_FAIL_PER_TASK}" && "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_SUCCESS_PER_TASK}" && "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi

"${PYTHON_BIN}" -m benchmark.real_world.examples.run_dummy \
    --fail-root      "${FAIL_ROOT}" \
    --success-root   "${SUCCESS_ROOT}" \
    --tasks          ${TASKS} \
    --save-json      "${SAVE_JSON}" \
    --proprio-field  "${PROPRIO_FIELD}" \
    --proprio-start  "${PROPRIO_START}" \
    --proprio-stop   "${PROPRIO_STOP}" \
    --action-start   "${ACTION_START}" \
    --action-stop    "${ACTION_STOP}" \
    --camera-name    "${CAMERA_NAME}" \
    --score-mode     "${SCORE_MODE}" \
    --seed           "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[real_world] wrote ${SAVE_JSON}"
