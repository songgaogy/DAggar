#!/usr/bin/env bash
# Build raw NPZ cache for the real-world Agilex benchmark.
# Run from repo root:
#   bash benchmark/real_world/scripts/build_cache.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

FAIL_ROOT="${FAIL_ROOT:-${REPO_ROOT}/data/agilex/failure_annotations/out_by_task}"
SUCCESS_ROOT="${SUCCESS_ROOT:-${REPO_ROOT}/data/agilex}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/data/.agilex_train_cache}"
TASKS="${TASKS:-candy_in_plate duck_in_bowl Micky_in_box sausage_in_pot}"

MAX_FAIL_PER_TASK="${MAX_FAIL_PER_TASK:-50}"
MAX_SUCCESS_PER_TASK="${MAX_SUCCESS_PER_TASK:-100}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/dagger/bin/python}"

ACTION_START="${ACTION_START:-7}"
ACTION_STOP="${ACTION_STOP:-14}"
PROPRIO_FIELD="${PROPRIO_FIELD:-qpos}"
PROPRIO_START="${PROPRIO_START:-7}"
PROPRIO_STOP="${PROPRIO_STOP:-14}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

EXTRA_ARGS=()
if [[ -n "${MAX_FAIL_PER_TASK}" && "${MAX_FAIL_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-per-task "${MAX_FAIL_PER_TASK}")
fi
if [[ -n "${MAX_SUCCESS_PER_TASK}" && "${MAX_SUCCESS_PER_TASK}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-per-task "${MAX_SUCCESS_PER_TASK}")
fi
if [[ -n "${CAMERAS:-}" ]]; then
    EXTRA_ARGS+=(--cameras ${CAMERAS})
fi

"${PYTHON_BIN}" -m benchmark.real_world.examples.build_cache \
    --fail-root      "${FAIL_ROOT}" \
    --success-root   "${SUCCESS_ROOT}" \
    --cache-root     "${CACHE_ROOT}" \
    --tasks          ${TASKS} \
    --proprio-field  "${PROPRIO_FIELD}" \
    --proprio-start  "${PROPRIO_START}" \
    --proprio-stop   "${PROPRIO_STOP}" \
    --action-start   "${ACTION_START}" \
    --action-stop    "${ACTION_STOP}" \
    --image-size     "${IMAGE_SIZE}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[real_world][cache] root ${CACHE_ROOT}"
