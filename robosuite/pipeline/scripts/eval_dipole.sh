#!/usr/bin/env bash
# Headless eval of a trained DIPOLE checkpoint.
#
# - Renders offscreen (no popping window) and saves .mp4 per episode under
#   <run_dir>/eval/<task>__<ckpt>__<omega>__<ts>/videos/ (run_dir inferred from CHECKPOINT).
# - Runs $EPISODES rollouts per omega and writes summary.json with success rate
#   (+ 95% Wilson CI), mean return, per-episode breakdown.
# - Optionally sweeps a list of omegas; one output dir per omega.
#
# Required:
#   CHECKPOINT=/abs/path/to/dipole/checkpoints/<step>.pt
#
# Optional (with defaults):
#   ENVIRONMENT=PickPlaceBread
#   TASK_NAME=$ENVIRONMENT
#   EPISODES=20
#   EPISODE_MAX_STEPS=300
#   OMEGA=2.0
#   OMEGA_SWEEP="${OMEGA}"           # comma-separated, e.g. "0,1,2,4"
#   VIDEO_OUTPUT=true                # set false to skip mp4 writing
#   VIDEO_CAMERA=agentview
#   VIDEO_FPS=20
#   VIDEO_IMAGE_SIZE=512
#   MAX_VIDEOS=0                     # cap saved videos per omega; <=0 = save all
#   OUTPUT_ROOT=                     # default: <run_dir>/eval (from CHECKPOINT path)
#   EVAL_DETERMINISTIC=false         # true => deterministic ODE init (no noise)
#   DEVICE=cuda:0
#   INIT_CHECKPOINT=                 # optional override (default picks from run_info.json)
#   SKIP_DIAGNOSTIC=false
#   STRICT_DIAGNOSTIC=false          # abort sweep if diagnostic verdict is BAD

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# ------------------------------------------------
CHECKPOINT="${CHECKPOINT:-outputs/DIPOLE/dipole_PickPlaceBread_2026-05-18_13-28-22/checkpoints/step_00010000_updates_00003754_ep_00032.pt}"
ENVIRONMENT="${ENVIRONMENT:-PickPlaceBread}"
TASK_NAME="${TASK_NAME:-${ENVIRONMENT}}"
EPISODES="${EPISODES:-50}"
EPISODE_MAX_STEPS="${EPISODE_MAX_STEPS:-400}"

OMEGA="${OMEGA:-1}"
OMEGA_SWEEP="${OMEGA_SWEEP:-${OMEGA}}"

VIDEO_OUTPUT="${VIDEO_OUTPUT:-true}"
VIDEO_CAMERA="${VIDEO_CAMERA:-agentview}"
VIDEO_FPS="${VIDEO_FPS:-20}"
VIDEO_IMAGE_SIZE="${VIDEO_IMAGE_SIZE:-512}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"

EVAL_DETERMINISTIC="${EVAL_DETERMINISTIC:-false}"
DEVICE="${DEVICE:-cuda:0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

SKIP_DIAGNOSTIC="${SKIP_DIAGNOSTIC:-false}"
STRICT_DIAGNOSTIC="${STRICT_DIAGNOSTIC:-false}"
# ------------------------------------------------

if [[ -z "${CHECKPOINT}" ]]; then
  echo "[ERROR] CHECKPOINT env var is required (path to a trained DIPOLE .pt or run dir)." >&2
  exit 1
fi
if [[ -d "${CHECKPOINT}" ]]; then
  if [[ -f "${CHECKPOINT}/checkpoints/latest.pt" ]]; then
    CHECKPOINT="${CHECKPOINT}/checkpoints/latest.pt"
  elif [[ -f "${CHECKPOINT}/latest.pt" ]]; then
    CHECKPOINT="${CHECKPOINT}/latest.pt"
  else
    echo "[ERROR] No latest.pt under directory ${CHECKPOINT}" >&2
    exit 1
  fi
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint file not found: ${CHECKPOINT}" >&2
  exit 1
fi

# Default eval artifacts under the training run dir: .../dipole_<task>_<ts>/eval/
RUN_DIR=""
if [[ "$(basename "$(dirname "${CHECKPOINT}")")" == "checkpoints" ]]; then
  RUN_DIR="$(cd "$(dirname "$(dirname "${CHECKPOINT}")")" && pwd)"
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  :
elif [[ -n "${RUN_DIR}" && -d "${RUN_DIR}" ]]; then
  OUTPUT_ROOT="${RUN_DIR}/eval"
else
  echo "[WARN] Could not infer run dir from CHECKPOINT; falling back to outputs/DIPOLE/eval." >&2
  OUTPUT_ROOT="${ROOT_DIR}/outputs/DIPOLE/eval"
fi

# Always force headless offscreen rendering. The python driver builds the env
# with has_renderer=False and uses sim.render(...) for offscreen video frames.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

cd "${ROOT_DIR}"

# --------------------------------------------------------------------------
# Step 1: cheap static branch-divergence diagnostic. Tells us up front whether
# pos/neg embeddings ever specialized; if they collapsed, omega sweep is moot.
# --------------------------------------------------------------------------
if [[ "${SKIP_DIAGNOSTIC}" != "true" ]]; then
  echo "==========================================================="
  echo "[eval_dipole] branch divergence diagnostic"
  echo "==========================================================="
  DIAG_ARGS=(--checkpoint "${CHECKPOINT}")
  if [[ "${STRICT_DIAGNOSTIC}" == "true" ]]; then
    DIAG_ARGS+=(--strict)
  fi
  if ! "${PYTHON_BIN}" -m robosuite.pipeline.utils.diagnose_dipole_divergence "${DIAG_ARGS[@]}"; then
    if [[ "${STRICT_DIAGNOSTIC}" == "true" ]]; then
      echo "[ERROR] Diagnostic returned BAD verdict under STRICT_DIAGNOSTIC=true. Aborting eval." >&2
      exit 1
    fi
    echo "[WARN] Diagnostic returned a non-zero exit; continuing anyway (set STRICT_DIAGNOSTIC=true to abort)." >&2
  fi
fi

# --------------------------------------------------------------------------
# Step 2: per-omega headless eval. Each omega gets its own output dir / mp4s /
# summary.json. At the end we aggregate success rates across the sweep.
# --------------------------------------------------------------------------
EXTRA_ARGS=("$@")
SUMMARY_DIRS=()

build_py_args() {
  local omega_value="$1"
  PY_ARGS=(
    --checkpoint "${CHECKPOINT}"
    --env-name "${ENVIRONMENT}"
    --task-name "${TASK_NAME}"
    --episodes "${EPISODES}"
    --episode-max-steps "${EPISODE_MAX_STEPS}"
    --omega "${omega_value}"
    --output-root "${OUTPUT_ROOT}"
    --video-output "${VIDEO_OUTPUT}"
    --video-camera "${VIDEO_CAMERA}"
    --video-fps "${VIDEO_FPS}"
    --video-height "${VIDEO_IMAGE_SIZE}"
    --video-width "${VIDEO_IMAGE_SIZE}"
    --max-videos "${MAX_VIDEOS}"
    --device "${DEVICE}"
  )
  if [[ "${EVAL_DETERMINISTIC}" == "true" ]]; then
    PY_ARGS+=(--deterministic)
  fi
  if [[ -n "${INIT_CHECKPOINT}" ]]; then
    PY_ARGS+=(--init-checkpoint "${INIT_CHECKPOINT}")
  fi
}

IFS=',' read -ra OMEGAS <<< "${OMEGA_SWEEP}"
for omega_raw in "${OMEGAS[@]}"; do
  omega="${omega_raw// /}"
  if [[ -z "${omega}" ]]; then
    continue
  fi
  echo "==========================================================="
  echo "[eval_dipole] omega=${omega}  episodes=${EPISODES}  output_root=${OUTPUT_ROOT}  ckpt=${CHECKPOINT}"
  echo "==========================================================="
  build_py_args "${omega}"
  RUN_OUTPUT_LOG="$(mktemp)"
  set +e
  "${PYTHON_BIN}" -m robosuite.pipeline.eval_dipole "${PY_ARGS[@]}" "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${RUN_OUTPUT_LOG}"
  status="${PIPESTATUS[0]}"
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "[ERROR] eval_dipole.py exited with status ${status} for omega=${omega}." >&2
    rm -f "${RUN_OUTPUT_LOG}"
    exit "${status}"
  fi
  run_dir="$(grep -E '^output_dir: ' "${RUN_OUTPUT_LOG}" | tail -n 1 | awk '{print $2}')"
  if [[ -n "${run_dir}" ]]; then
    SUMMARY_DIRS+=("${run_dir}")
  fi
  rm -f "${RUN_OUTPUT_LOG}"
done

# --------------------------------------------------------------------------
# Step 3: sweep-level aggregate summary. Reads every per-omega summary.json
# and prints a single table; also writes <run_dir>/eval/sweep_summary.json when
# the sweep had > 1 entry.
# --------------------------------------------------------------------------
if [[ "${#SUMMARY_DIRS[@]}" -gt 0 ]]; then
  echo "==========================================================="
  echo "[eval_dipole] sweep summary"
  echo "==========================================================="
  "${PYTHON_BIN}" - "${SUMMARY_DIRS[@]}" <<'PY'
import json
import sys
from pathlib import Path

rows = []
for run_dir in sys.argv[1:]:
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        continue
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows.append(data)

if not rows:
    print("[sweep] no summary.json files found.")
    sys.exit(0)

rows.sort(key=lambda row: float(row.get("omega", 0.0)))
print(f"{'omega':>7}  {'episodes':>9}  {'succ':>5}  {'succ_rate':>10}  {'ci95':>16}  {'mean_steps':>11}  {'mean_return':>12}")
for row in rows:
    omega = float(row.get("omega", 0.0))
    n = int(row.get("episodes", 0))
    succ = int(row.get("success_count", 0))
    rate = float(row.get("success_rate", 0.0))
    lo = float(row.get("success_rate_ci95_low", 0.0))
    hi = float(row.get("success_rate_ci95_high", 0.0))
    mean_steps = float(row.get("mean_steps", 0.0))
    mean_return = float(row.get("mean_return", 0.0))
    print(f"{omega:7.3f}  {n:9d}  {succ:5d}  {rate:10.3f}  [{lo:0.2f}, {hi:0.2f}]  {mean_steps:11.1f}  {mean_return:12.3f}")

if len(rows) > 1:
    first_dir = Path(rows[0]["output_dir"]).parent
    out_path = first_dir / "sweep_summary.json"
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[sweep] wrote {out_path}")
PY
fi

# Examples:
#
# 1) Single omega, headless, 20 episodes, save all videos:
#    CHECKPOINT=outputs/DIPOLE/<run>/checkpoints/latest.pt \
#    OMEGA=2 EPISODES=20 \
#    bash robosuite/pipeline/scripts/eval_dipole.sh
#
# 2) Omega sweep with 30 episodes each:
#    CHECKPOINT=... OMEGA_SWEEP="0,1,2,4" EPISODES=30 \
#    bash robosuite/pipeline/scripts/eval_dipole.sh
#
# 3) Quick smoke run, no videos, abort if branches collapsed:
#    CHECKPOINT=... OMEGA_SWEEP="0,2,4" EPISODES=5 \
#    VIDEO_OUTPUT=false STRICT_DIAGNOSTIC=true \
#    bash robosuite/pipeline/scripts/eval_dipole.sh
