#!/usr/bin/env bash
# D4-Disc conditional dynamics trainer entry point.
# Run from repo root:
#   bash robosuite/discriminator/d4disc/scripts/train_d4.sh
# Overrides via env vars, e.g.:
#   WARM_UP_EPOCHS=4 BOOTSTRAP_EPOCHS=20 TASKS="PickPlaceBread" \
#     bash robosuite/discriminator/d4disc/scripts/train_d4.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"
POLICY_CKPT="${POLICY_CKPT:-${REPO_ROOT}/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/data/.lpb_score_cache}"

if [[ ! -f "${POLICY_CKPT}" ]]; then
    echo "[d4_train] ERROR: POLICY_CKPT not found: ${POLICY_CKPT}" >&2
    exit 1
fi

TASKS="${TASKS:-PandaLift PandaPickPlaceCan PandaStack PickPlaceBread PickPlaceCereal PickPlaceMilk}"

EXPERT_NUM="${EXPERT_NUM:-0}"
SUCC_NUM="${SUCC_NUM:-200}"
FAIL_NUM="${FAIL_NUM:-100}"

EXPERT_PATHS=()
ROLLOUT_PATHS=()
FAIL_PATHS=()
for task in ${TASKS}; do
    ep="${REPO_ROOT}/data/${task}/expert"
    rp="${REPO_ROOT}/data/${task}/success_rollout"
    fp="${REPO_ROOT}/data/${task}/fail_rollout"
    if [[ -d "${ep}" ]]; then EXPERT_PATHS+=("${ep}"); fi
    if [[ -d "${rp}" ]]; then ROLLOUT_PATHS+=("${rp}"); fi
    if [[ -d "${fp}" ]]; then FAIL_PATHS+=("${fp}"); fi
done
if [[ ${#EXPERT_PATHS[@]} -eq 0 && ${#ROLLOUT_PATHS[@]} -eq 0 && ${#FAIL_PATHS[@]} -eq 0 ]]; then
    echo "[d4_train] ERROR: no expert/success/fail rollout directories found for TASKS=${TASKS}" >&2
    exit 1
fi

# Dataset.
HORIZON="${HORIZON:-1}"

# Architecture.
D_MODEL="${D_MODEL:-512}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-4}"
DROPOUT="${DROPOUT:-0.1}"
MAX_ACTION_HORIZON="${MAX_ACTION_HORIZON:-32}"
D_COND="${D_COND:-64}"
# AdaLN modulation init std. 0.0 => DiT AdaLN-Zero (identity at step 0;
# safe but c=- branch can take many epochs to differentiate from c=+).
# Default 0.02: breaks f(+)=f(-) degeneracy at init so the fail-data gradient
# through the c=- branch grows faster during bootstrap. Set 0.0 for pure
# AdaLN-Zero behavior.
ADALN_INIT_STD="${ADALN_INIT_STD:-0.05}"

# Phase schedule.
WARM_UP_EPOCHS="${WARM_UP_EPOCHS:-10}"
BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-40}"

# Training.
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-3e-4}"
WD="${WD:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
PROPRIO_LOSS_WEIGHT="${PROPRIO_LOSS_WEIGHT:-0.1}"
LATENT_LOSS_WEIGHT="${LATENT_LOSS_WEIGHT:-1.0}"
SIGMA_SQ="${SIGMA_SQ:-0.5}"
USE_BF16="${USE_BF16:-0}"
LOG_EVERY="${LOG_EVERY:-50}"

# CFG.
P_COND_DROP="${P_COND_DROP:-0.1}"
MIN_BATCH_BALANCE="${MIN_BATCH_BALANCE:-0.15}"

# Schedule.
ALPHA_EXPONENT="${ALPHA_EXPONENT:-0.75}"
# alpha_cap_rate default bumped 1.0 -> 5.0 after run d4dyn_20260422_050952
# showed alpha pinned at ~1/held_out_r_plus and gamma stuck at 0.5.
ALPHA_CAP_RATE="${ALPHA_CAP_RATE:-5.0}"
ETA_FINAL="${ETA_FINAL:-0.3}"
ETA_RAMP_EPOCHS="${ETA_RAMP_EPOCHS:-5}"

# F3 gamma warm-start at end of Phase A (breaks gamma=0.5 symmetric fixed
# point using KNN distance to the clean success bank).
F3_WARM_START="${F3_WARM_START:-1}"
F3_WARM_START_K="${F3_WARM_START_K:-1}"
F3_WARM_START_MAX_CLEAN="${F3_WARM_START_MAX_CLEAN:-100000}"
# 'rank' (default, uniform quantile — robust to scale mismatch between clean
# self-kNN and fail→clean distance distributions that caused the v2 run to
# saturate at γ=0.999 for every sample) or 'sigmoid' (auto β/κ on fail→clean).
WARM_START_MODE="${WARM_START_MODE:-rank}"
# Skip gate updates for the first N bootstrap epochs after warm-start so
# f(-) has time to differentiate before the advantage gate overrides gamma.
GATE_FREEZE_EPOCHS="${GATE_FREEZE_EPOCHS:-4}"
# Skip Phase B entirely: ship Phase-A-only f(+) as the detector (use OMEGA=0
# at benchmark time). This is the "pos-branch-only learned-dynamics residual"
# baseline. Set SKIP_BOOTSTRAP=1 to enable.
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"

# Stability.
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_ALPHA_GAMMA="${EMA_ALPHA_GAMMA:-0.5}"
GAMMA_CLAMP="${GAMMA_CLAMP:-1e-3}"
HELD_OUT_RATIO="${HELD_OUT_RATIO:-0.05}"

SEED="${SEED:-0}"

MAX_TRAJ_PER_KIND="${MAX_TRAJ_PER_KIND:-0}"
if [[ "${MAX_TRAJ_PER_KIND}" -le 0 ]]; then
    if [[ "${EXPERT_NUM}" -gt 0 || "${SUCC_NUM}" -gt 0 || "${FAIL_NUM}" -gt 0 ]]; then
        MAX_TRAJ_PER_KIND=0
        for n in "${EXPERT_NUM}" "${SUCC_NUM}" "${FAIL_NUM}"; do
            if [[ "${n}" -gt "${MAX_TRAJ_PER_KIND}" ]]; then MAX_TRAJ_PER_KIND="${n}"; fi
        done
    fi
fi

RUN_NAME="${RUN_NAME:-d4dyn_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/d4disc/dynamics/${RUN_NAME}}"
SAVE_NAME="${SAVE_NAME:-d4_dynamics.pt}"
SAVE_FREQ="${SAVE_FREQ:-10}"
mkdir -p "${SAVE_DIR}"
LOG_PATH="${LOG_PATH:-${SAVE_DIR}/train.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1
echo "[d4_train] logging to ${LOG_PATH}"

EXTRA_ARGS=()
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    EXTRA_ARGS+=(--proprio-indices ${PROPRIO_INDICES})
fi
if [[ "${MAX_TRAJ_PER_KIND}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-trajectories-per-kind "${MAX_TRAJ_PER_KIND}")
fi
if [[ "${USE_BF16}" == "1" ]]; then
    EXTRA_ARGS+=(--use-bfloat16)
fi
if [[ "${DISABLE_WANDB:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--disable-wandb)
fi
if [[ "${F3_WARM_START}" == "1" ]]; then
    EXTRA_ARGS+=(--f3-warm-start)
else
    EXTRA_ARGS+=(--no-f3-warm-start)
fi
EXTRA_ARGS+=(--f3-warm-start-k "${F3_WARM_START_K}")
EXTRA_ARGS+=(--f3-warm-start-max-clean "${F3_WARM_START_MAX_CLEAN}")
EXTRA_ARGS+=(--warm-start-mode "${WARM_START_MODE}")
EXTRA_ARGS+=(--gate-freeze-epochs "${GATE_FREEZE_EPOCHS}")
if [[ "${SKIP_BOOTSTRAP}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-bootstrap)
fi

EXPERT_ARG=()
ROLLOUT_ARG=()
FAIL_ARG=()
if [[ ${#EXPERT_PATHS[@]} -gt 0 ]]; then EXPERT_ARG+=(--expert-paths "${EXPERT_PATHS[@]}"); fi
if [[ ${#ROLLOUT_PATHS[@]} -gt 0 ]]; then ROLLOUT_ARG+=(--rollout-paths "${ROLLOUT_PATHS[@]}"); fi
if [[ ${#FAIL_PATHS[@]} -gt 0 ]]; then FAIL_ARG+=(--fail-paths "${FAIL_PATHS[@]}"); fi

WANDB_MODE="${WANDB_MODE:-offline}" WANDB_NAME="${WANDB_NAME:-songgao-personal}" \
"${PYTHON_BIN}" -m robosuite.discriminator.d4disc.train_d4 \
    --policy-ckpt              "${POLICY_CKPT}" \
    --cache-root               "${CACHE_ROOT}" \
    "${EXPERT_ARG[@]}" \
    "${ROLLOUT_ARG[@]}" \
    "${FAIL_ARG[@]}" \
    --horizon                  "${HORIZON}" \
    --d-model                  "${D_MODEL}" \
    --num-layers               "${NUM_LAYERS}" \
    --num-heads                "${NUM_HEADS}" \
    --dropout                  "${DROPOUT}" \
    --max-action-horizon       "${MAX_ACTION_HORIZON}" \
    --d-cond                   "${D_COND}" \
    --adaln-init-std           "${ADALN_INIT_STD}" \
    --warm-up-epochs           "${WARM_UP_EPOCHS}" \
    --bootstrap-epochs         "${BOOTSTRAP_EPOCHS}" \
    --device                   "${DEVICE}" \
    --batch-size               "${BATCH_SIZE}" \
    --num-workers              "${NUM_WORKERS}" \
    --lr                       "${LR}" \
    --weight-decay             "${WD}" \
    --grad-clip-norm           "${GRAD_CLIP}" \
    --proprio-loss-weight      "${PROPRIO_LOSS_WEIGHT}" \
    --latent-loss-weight       "${LATENT_LOSS_WEIGHT}" \
    --sigma-sq                 "${SIGMA_SQ}" \
    --log-every                "${LOG_EVERY}" \
    --p-cond-drop              "${P_COND_DROP}" \
    --min-batch-balance        "${MIN_BATCH_BALANCE}" \
    --alpha-exponent           "${ALPHA_EXPONENT}" \
    --alpha-cap-rate           "${ALPHA_CAP_RATE}" \
    --eta-final                "${ETA_FINAL}" \
    --eta-ramp-epochs          "${ETA_RAMP_EPOCHS}" \
    --ema-decay                "${EMA_DECAY}" \
    --ema-alpha-gamma          "${EMA_ALPHA_GAMMA}" \
    --gamma-clamp              "${GAMMA_CLAMP}" \
    --held-out-ratio           "${HELD_OUT_RATIO}" \
    --run-name                 "${RUN_NAME}" \
    --seed                     "${SEED}" \
    --save-dir                 "${SAVE_DIR}" \
    --save-name                "${SAVE_NAME}" \
    --save-freq                "${SAVE_FREQ}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

echo "[d4_train] wrote checkpoints under ${SAVE_DIR}"
