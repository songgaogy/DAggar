#!/usr/bin/env bash
# D4-Disc cached-image trainer entry point.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"
PREPROCESSED_CACHE_ROOT="${PREPROCESSED_CACHE_ROOT:-${REPO_ROOT}/data/.lpb_score_preprocessed_cache}"

if [[ ! -d "${PREPROCESSED_CACHE_ROOT}" ]]; then
    echo "[d4_train] ERROR: PREPROCESSED_CACHE_ROOT not found: ${PREPROCESSED_CACHE_ROOT}" >&2
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

HORIZON="${HORIZON:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"

ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-}"
ENCODER_PRETRAINED="${ENCODER_PRETRAINED:-0}"
ENCODER_FREEZE="${ENCODER_FREEZE:-0}"
ENCODER_LR_MULT="${ENCODER_LR_MULT:-0.1}"

D_MODEL="${D_MODEL:-512}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-8}"
DROPOUT="${DROPOUT:-0.1}"
MAX_ACTION_HORIZON="${MAX_ACTION_HORIZON:-32}"
D_COND="${D_COND:-64}"
ADALN_INIT_STD="${ADALN_INIT_STD:-0.05}"

WARM_UP_EPOCHS="${WARM_UP_EPOCHS:-10}"
BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-40}"

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

MIN_BATCH_BALANCE="${MIN_BATCH_BALANCE:-0.15}"

ALPHA_EXPONENT="${ALPHA_EXPONENT:-0.75}"
ALPHA_CAP_RATE="${ALPHA_CAP_RATE:-5.0}"
ETA_FINAL="${ETA_FINAL:-0.3}"
ETA_RAMP_EPOCHS="${ETA_RAMP_EPOCHS:-5}"

F3_WARM_START="${F3_WARM_START:-1}"
F3_WARM_START_K="${F3_WARM_START_K:-1}"
F3_WARM_START_MAX_CLEAN="${F3_WARM_START_MAX_CLEAN:-100000}"
WARM_START_MODE="${WARM_START_MODE:-rank}"
GATE_FREEZE_EPOCHS="${GATE_FREEZE_EPOCHS:-4}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"

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
if [[ "${ENCODER_PRETRAINED}" == "1" ]]; then
    EXTRA_ARGS+=(--encoder-pretrained)
else
    EXTRA_ARGS+=(--no-encoder-pretrained)
fi
if [[ "${ENCODER_FREEZE}" == "1" ]]; then
    EXTRA_ARGS+=(--encoder-freeze)
else
    EXTRA_ARGS+=(--no-encoder-freeze)
fi
if [[ -n "${ENCODER_CHECKPOINT}" ]]; then
    EXTRA_ARGS+=(--encoder-checkpoint "${ENCODER_CHECKPOINT}")
fi
EXTRA_ARGS+=(--encoder-lr-mult "${ENCODER_LR_MULT}")
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
    --preprocessed-cache-root  "${PREPROCESSED_CACHE_ROOT}" \
    "${EXPERT_ARG[@]}" \
    "${ROLLOUT_ARG[@]}" \
    "${FAIL_ARG[@]}" \
    --horizon                  "${HORIZON}" \
    --image-size               "${IMAGE_SIZE}" \
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
