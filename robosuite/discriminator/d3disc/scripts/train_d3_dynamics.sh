#!/usr/bin/env bash
# D3-Disc latent-dynamics trainer entry point.
# Run from repo root:
#   bash robosuite/discriminator/d3disc/scripts/train_d3_dynamics.sh
# Overrides via env vars, e.g.:
#   EPOCHS=40 SUCC_NUM=400 EXPERT_NUM=0 \
#     bash robosuite/discriminator/d3disc/scripts/train_d3_dynamics.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"
POLICY_CKPT="${POLICY_CKPT:-${REPO_ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/data/.lpb_score_cache}"

if [[ ! -f "${POLICY_CKPT}" ]]; then
    echo "[d3_train] ERROR: POLICY_CKPT not found: ${POLICY_CKPT}" >&2
    exit 1
fi

# Default tasks span the 6 canonical envs.
TASKS="${TASKS:-PandaLift PandaPickPlaceCan PandaStack PickPlaceBread PickPlaceCereal PickPlaceMilk}"

# Per-task trajectory caps for expert + success_rollout directories.
# EXPERT_NUM / SUCC_NUM map to --max-trajectories-per-kind (applied per HDF5 file
# within each path). Set to 0 to disable the cap.
EXPERT_NUM="${EXPERT_NUM:-0}"     # 0 = all
SUCC_NUM=200         # 0 = all (this is the "train-time SUCC_NUM")

# Build expert / rollout path lists from TASKS.
EXPERT_PATHS=()
ROLLOUT_PATHS=()
for task in ${TASKS}; do
    ep="${REPO_ROOT}/data/${task}/expert"
    rp="${REPO_ROOT}/data/${task}/success_rollout"
    if [[ -d "${ep}" ]]; then EXPERT_PATHS+=("${ep}"); fi
    if [[ -d "${rp}" ]]; then ROLLOUT_PATHS+=("${rp}"); fi
done
if [[ ${#EXPERT_PATHS[@]} -eq 0 && ${#ROLLOUT_PATHS[@]} -eq 0 ]]; then
    echo "[d3_train] ERROR: no expert/success_rollout directories found for TASKS=${TASKS}" >&2
    exit 1
fi

# Dataset.
HORIZON="${HORIZON:-1}"

# Architecture (LPB-compatible defaults).
D_MODEL="${D_MODEL:-512}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
MAX_ACTION_HORIZON="${MAX_ACTION_HORIZON:-32}"

# Training.
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-3e-4}"
WD="${WD:-1e-4}"
EXPERT_RATIO="${EXPERT_RATIO:-0.5}"
PROPRIO_LOSS_WEIGHT="${PROPRIO_LOSS_WEIGHT:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
LOG_EVERY="${LOG_EVERY:-50}"
VAL_RATIO="${VAL_RATIO:-0.05}"
SEED="${SEED:-0}"

# Use the LARGER of EXPERT_NUM / SUCC_NUM — the dataset's
# max_trajectories_per_kind applies to each kind independently.
MAX_TRAJ_PER_KIND="${MAX_TRAJ_PER_KIND:-0}"
if [[ "${MAX_TRAJ_PER_KIND}" -le 0 ]]; then
    if [[ "${EXPERT_NUM}" -gt 0 || "${SUCC_NUM}" -gt 0 ]]; then
        MAX_TRAJ_PER_KIND=$(( EXPERT_NUM > SUCC_NUM ? EXPERT_NUM : SUCC_NUM ))
    fi
fi

RUN_NAME="${RUN_NAME:-d3dyn_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/d3disc/dynamics/${RUN_NAME}}"
SAVE_NAME="${SAVE_NAME:-d3_dynamics.pt}"
SAVE_FREQ="${SAVE_FREQ:-10}"
mkdir -p "${SAVE_DIR}"

EXTRA_ARGS=()
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    EXTRA_ARGS+=(--proprio-indices ${PROPRIO_INDICES})
fi
if [[ "${MAX_TRAJ_PER_KIND}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-trajectories-per-kind "${MAX_TRAJ_PER_KIND}")
fi

"${PYTHON_BIN}" -m robosuite.discriminator.d3disc.train_dynamics \
    --policy-ckpt              "${POLICY_CKPT}" \
    --cache-root               "${CACHE_ROOT}" \
    --expert-paths             "${EXPERT_PATHS[@]}" \
    --rollout-paths            "${ROLLOUT_PATHS[@]}" \
    --horizon                  "${HORIZON}" \
    --d-model                  "${D_MODEL}" \
    --num-layers               "${NUM_LAYERS}" \
    --num-heads                "${NUM_HEADS}" \
    --dropout                  "${DROPOUT}" \
    --max-action-horizon       "${MAX_ACTION_HORIZON}" \
    --device                   "${DEVICE}" \
    --epochs                   "${EPOCHS}" \
    --batch-size               "${BATCH_SIZE}" \
    --num-workers              "${NUM_WORKERS}" \
    --lr                       "${LR}" \
    --weight-decay             "${WD}" \
    --expert-ratio             "${EXPERT_RATIO}" \
    --proprio-loss-weight      "${PROPRIO_LOSS_WEIGHT}" \
    --grad-clip-norm           "${GRAD_CLIP}" \
    --log-every                "${LOG_EVERY}" \
    --val-ratio                "${VAL_RATIO}" \
    --seed                     "${SEED}" \
    --save-dir                 "${SAVE_DIR}" \
    --save-name                "${SAVE_NAME}" \
    --save-freq                "${SAVE_FREQ}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[d3_train] wrote checkpoints under ${SAVE_DIR}"
