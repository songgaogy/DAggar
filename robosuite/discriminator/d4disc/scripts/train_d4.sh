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

TASKS="PandaLift PandaPickPlaceCan PandaStack PickPlaceBread PickPlaceCereal PickPlaceMilk"

# ------------------------------
# data
EXPERT_NUM="${EXPERT_NUM:-0}"
SUCC_NUM="${SUCC_NUM:-100}"
FAIL_NUM="${FAIL_NUM:-100}"

WARM_UP_EPOCHS="${WARM_UP_EPOCHS:-10}"
BOOTSTRAP_EPOCHS="${BOOTSTRAP_EPOCHS:-40}"

F3_WARM_START="${F3_WARM_START:-1}"          # 1: KNN-based gamma warm-start at end of Phase A (recommended)
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"        # 1: run Phase A only (pos-branch baseline; use omega=0 at eval)
F3_WARM_START_K="${F3_WARM_START_K:-1}"                  # KNN order (k=1 is 1-NN)

# Phase-B advantage gate.
#   "knn"      (default): score each conditional prediction against an expert
#                         z_{t+h} bank; advantage = log p_+ - log p_-. Matches
#                         inference-time score_mode=knn. Removes the motion-
#                         magnitude confound of raw residual scoring.
#   "residual"          : legacy ||f(c) - z_target||² (d4disc_0423.md §5).
ADVANTAGE_MODE="${ADVANTAGE_MODE:-knn}"
ADVANTAGE_KNN_BANK_SIZE="${ADVANTAGE_KNN_BANK_SIZE:-100000}"
ADVANTAGE_KNN_CHUNK_SIZE="${ADVANTAGE_KNN_CHUNK_SIZE:-8192}"
ADVANTAGE_KNN_K="${ADVANTAGE_KNN_K:-1}"

# Phase-B M-step repel loss. Pushes c=- branch away from expert bank B_+.
# Off by default; recommended first try REPEL_WEIGHT=0.1.
REPEL_WEIGHT="${REPEL_WEIGHT:-0.05}"
REPEL_MARGIN="${REPEL_MARGIN:--1.0}"
REPEL_MARGIN_PERCENTILE="${REPEL_MARGIN_PERCENTILE:-0.5}"
REPEL_ON_PHASE_A="${REPEL_ON_PHASE_A:-0}"
REPEL_WARMUP_EPOCHS="${REPEL_WARMUP_EPOCHS:-3}"

# Training hyperparameters
DEVICE="${DEVICE:-cuda:1}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-3e-4}"
WD="${WD:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
PROPRIO_LOSS_WEIGHT="${PROPRIO_LOSS_WEIGHT:-0.1}"
LATENT_LOSS_WEIGHT="${LATENT_LOSS_WEIGHT:-1.0}"
SIGMA_SQ="${SIGMA_SQ:-0.5}"
USE_BF16="${USE_BF16:-0}"
LOG_EVERY="${LOG_EVERY:-50}"
# ------------------------------

EXPERT_PATHS=()
ROLLOUT_PATHS=()
FAIL_PATHS=()
for task in ${TASKS}; do
    ep="${REPO_ROOT}/data/${task}/expert"
    rp="${REPO_ROOT}/data/${task}/success_rollout"
    fp="${REPO_ROOT}/data/${task}/fail_rollout"
    if [[ "${EXPERT_NUM}" -gt 0 && -d "${ep}" ]]; then EXPERT_PATHS+=("${ep}"); fi
    if [[ "${SUCC_NUM}" -gt 0 && -d "${rp}" ]]; then ROLLOUT_PATHS+=("${rp}"); fi
    if [[ "${FAIL_NUM}" -gt 0 && -d "${fp}" ]]; then FAIL_PATHS+=("${fp}"); fi
done
if [[ ${#EXPERT_PATHS[@]} -eq 0 && ${#ROLLOUT_PATHS[@]} -eq 0 && ${#FAIL_PATHS[@]} -eq 0 ]]; then
    echo "[d4_train] ERROR: no data roots selected (check EXPERT_NUM/SUCC_NUM/FAIL_NUM and TASKS=${TASKS})" >&2
    exit 1
fi

HORIZON="${HORIZON:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"

# Encoder defaults mirror lpb/train_lpb_dynamics.sh to avoid
# representation collapse: ImageNet-pretrained ResNet-18 (loaded from the same
# checkpoint LPB uses), frozen during training. With a trainable + randomly
# initialized encoder, `pred_latent = z_t + Delta(...)` trivially minimizes by
# making all embeddings identical (latent_mse -> ~1e-6 but AUROC collapses to
# chance). Override via env vars for encoder-training ablations.
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-${REPO_ROOT}/robosuite/RL/models/resnet18-f37072fd.pth}"
ENCODER_PRETRAINED="${ENCODER_PRETRAINED:-1}"
ENCODER_FREEZE="${ENCODER_FREEZE:-1}"
ENCODER_LR_MULT="${ENCODER_LR_MULT:-0.1}"

D_MODEL="${D_MODEL:-512}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-8}"
DROPOUT="${DROPOUT:-0.1}"
MAX_ACTION_HORIZON="${MAX_ACTION_HORIZON:-32}"
D_COND="${D_COND:-64}"
ADALN_INIT_STD="${ADALN_INIT_STD:-0.05}"

MIN_BATCH_BALANCE="${MIN_BATCH_BALANCE:-0.15}"

ALPHA_EXPONENT="${ALPHA_EXPONENT:-0.75}"
ALPHA_CAP_RATE="${ALPHA_CAP_RATE:-5.0}"
ETA_FINAL="${ETA_FINAL:-0.3}"
ETA_RAMP_EPOCHS="${ETA_RAMP_EPOCHS:-5}"

# F3 warm-start (Phase A -> Phase B): initialize gamma on fail_raw samples from
# a D3-style KNN distance to the clean bank, to break the gamma≈0.5 symmetric
# fixed point early in bootstrap.
F3_WARM_START_MAX_CLEAN="${F3_WARM_START_MAX_CLEAN:-100000}"  # subsample clean bank cap
WARM_START_MODE="${WARM_START_MODE:-rank}"              # {'rank','sigmoid'} mapping d2 -> gamma
GATE_FREEZE_EPOCHS="${GATE_FREEZE_EPOCHS:-4}"           # hold gamma fixed for first N bootstrap epochs

# EMA + stability: EMA weights are used both for the advantage gate and for the
# inference-time checkpoint payload.
EMA_DECAY="${EMA_DECAY:-0.999}"                         # predictor weight EMA decay
EMA_ALPHA_GAMMA="${EMA_ALPHA_GAMMA:-0.5}"               # gamma_buffer EMA smoothing (higher = slower updates)
GAMMA_CLAMP="${GAMMA_CLAMP:-1e-3}"                      # clamp gamma to [eps, 1-eps] to avoid hard 0/1 routing
HELD_OUT_RATIO="${HELD_OUT_RATIO:-0.05}"                # fraction of clean positives held out to monitor r_plus

SEED="${SEED:-0}"

RUN_NAME="${RUN_NAME:-d4dyn_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints/d4disc/dynamics/${RUN_NAME}}"
SAVE_NAME="${SAVE_NAME:-d4_dynamics.pt}"
SAVE_FREQ="${SAVE_FREQ:-10}"
mkdir -p "${SAVE_DIR}"
LOG_PATH="${LOG_PATH:-${SAVE_DIR}/train.log}"
exec > >(tee -a "${LOG_PATH}") 2>&1
echo "[d4_train] logging to ${LOG_PATH}"
echo "[d4_train] per-task trajectory caps: expert=${EXPERT_NUM} success_rollout=${SUCC_NUM} fail_rollout=${FAIL_NUM} (0=disable kind)"

EXTRA_ARGS=()
if [[ -n "${PROPRIO_INDICES:-}" ]]; then
    EXTRA_ARGS+=(--proprio-indices ${PROPRIO_INDICES})
fi
if [[ "${EXPERT_NUM}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-expert-trajectories "${EXPERT_NUM}")
fi
if [[ "${SUCC_NUM}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-success-rollout-trajectories "${SUCC_NUM}")
fi
if [[ "${FAIL_NUM}" -gt 0 ]]; then
    EXTRA_ARGS+=(--max-fail-rollout-trajectories "${FAIL_NUM}")
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
EXTRA_ARGS+=(--advantage-mode "${ADVANTAGE_MODE}")
EXTRA_ARGS+=(--advantage-knn-bank-size "${ADVANTAGE_KNN_BANK_SIZE}")
EXTRA_ARGS+=(--advantage-knn-chunk-size "${ADVANTAGE_KNN_CHUNK_SIZE}")
EXTRA_ARGS+=(--advantage-knn-k "${ADVANTAGE_KNN_K}")
EXTRA_ARGS+=(--repel-weight "${REPEL_WEIGHT}")
EXTRA_ARGS+=(--repel-margin "${REPEL_MARGIN}")
EXTRA_ARGS+=(--repel-margin-percentile "${REPEL_MARGIN_PERCENTILE}")
EXTRA_ARGS+=(--repel-warmup-epochs "${REPEL_WARMUP_EPOCHS}")
if [[ "${REPEL_ON_PHASE_A}" == "1" ]]; then
    EXTRA_ARGS+=(--repel-on-phase-a)
fi
if [[ "${SKIP_BOOTSTRAP}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-bootstrap)
fi

EXPERT_ARG=()
ROLLOUT_ARG=()
FAIL_ARG=()
if [[ ${#EXPERT_PATHS[@]} -gt 0 ]]; then EXPERT_ARG+=(--expert-paths "${EXPERT_PATHS[@]}"); fi
if [[ ${#ROLLOUT_PATHS[@]} -gt 0 ]]; then ROLLOUT_ARG+=(--rollout-paths "${ROLLOUT_PATHS[@]}"); fi
if [[ ${#FAIL_PATHS[@]} -gt 0 ]]; then FAIL_ARG+=(--fail-paths "${FAIL_PATHS[@]}"); fi

PYTHONUNBUFFERED=1 WANDB_MODE="${WANDB_MODE:-offline}" WANDB_NAME="${WANDB_NAME:-songgao-personal}" \
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
