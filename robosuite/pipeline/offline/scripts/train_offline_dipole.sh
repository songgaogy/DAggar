set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# -------------------------------------
TASK="${TASK:-PickPlaceCereal}"
POLICY_CKPT="${POLICY_CKPT:-checkpoints/multitask_6/flow_multi_ep0100.pt}"
NNPU_CKPT="${NNPU_CKPT:-checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth}"
VAST_CKPT_EXPLICIT=0
if [[ -n "${VAST_CKPT:-}" || -n "${IQL_CKPT:-}" ]]; then
  VAST_CKPT_EXPLICIT=1
fi
if [[ -n "${VAST_CKPT:-}" && -n "${IQL_CKPT:-}" ]]; then
  echo "[ERROR] VAST_CKPT and deprecated IQL_CKPT conflict; set only VAST_CKPT." >&2
  exit 1
fi
if [[ -z "${VAST_CKPT:-}" && -n "${IQL_CKPT:-}" ]]; then
  echo "[WARN] IQL_CKPT is deprecated; use VAST_CKPT." >&2
  VAST_CKPT="${IQL_CKPT}"
fi
VAST_CKPT="${VAST_CKPT:-outputs/dipole_rl-vast/baseline/${TASK}/vast_state.pt}"
if [[ "${VAST_CKPT_EXPLICIT}" == "0" && ! -f "${VAST_CKPT}" ]]; then
  LEGACY_CKPT="outputs/dipole_rl-vast/baseline/${TASK}/iql_state.pt"
  if [[ -f "${LEGACY_CKPT}" ]]; then
    echo "[WARN] Loading deprecated checkpoint ${LEGACY_CKPT}; use vast_state.pt." >&2
    VAST_CKPT="${LEGACY_CKPT}"
  fi
fi
OFFLINE_EPISODES="${OFFLINE_EPISODES:-data/${TASK}/offline_data/offline_episodes.pt}"
PRETRAIN_DATA="${PRETRAIN_DATA:-data/${TASK}/pretrain_data-20260615_174814}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-15000}"
VAST_FINETUNE_STEPS="${VAST_FINETUNE_STEPS:-10000}"
VAST_BATCH_SIZE="${VAST_BATCH_SIZE:-512}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-256}"
VAST_V_MODE="${VAST_V_MODE:-single_vast}"
VAST_MAX_K="${VAST_MAX_K:-10}"
VAST_COMP_COEF="${VAST_COMP_COEF:-0.5}"
VAST_SAMPLING_SEED="${VAST_SAMPLING_SEED:-${SEED}}"
EXPECTILE_TAU="${EXPECTILE_TAU:-0.9}"
G_LR="${G_LR:-1.0e-4}"
V_LR="${V_LR:-1.0e-4}"
OUTPUT_REWARD_COEF="${OUTPUT_REWARD_COEF:-1}"
DISC_REWARD_COEF="${DISC_REWARD_COEF:-0}"

# w_pos = sigmoid(beta * (G + k)); decision boundary at G=-k.
BRANCH_BETA="${BRANCH_BETA:-1}"
BRANCH_K="${BRANCH_K:--2}"
RUN_SUBFIX="${RUN_SUBFIX:-beta${BRANCH_BETA}_k${BRANCH_K}_vast-${VAST_V_MODE}-K${VAST_MAX_K}}"

# SKIP_RL=1: skip Phase A VAST finetune and reuse VAST_FINETUNED.
SKIP_RL="${SKIP_RL:-0}"
if [[ -n "${VAST_FINETUNED:-}" && -n "${IQL_FINETUNED:-}" ]]; then
  echo "[ERROR] VAST_FINETUNED and deprecated IQL_FINETUNED conflict; set only VAST_FINETUNED." >&2
  exit 1
fi
if [[ -z "${VAST_FINETUNED:-}" && -n "${IQL_FINETUNED:-}" ]]; then
  echo "[WARN] IQL_FINETUNED is deprecated; use VAST_FINETUNED." >&2
  VAST_FINETUNED="${IQL_FINETUNED}"
fi
VAST_FINETUNED="${VAST_FINETUNED:-}"
USE_ONLINE_SUCCESS="${USE_ONLINE_SUCCESS:-0}"
# -------------------------------------

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1
# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

HYDRA_OVERRIDES=(
  "seed=${SEED}"
  "env.environment=${TASK}"
  "runtime.init_checkpoint=${POLICY_CKPT}"
  "algorithm.discriminator.checkpoint=${NNPU_CKPT}"
  "algorithm.vast.warmup_ckpt=${VAST_CKPT}"
  "algorithm.vast.config.device=${DEVICE}"
  "algorithm.vast.config.vast_v_mode=${VAST_V_MODE}"
  "algorithm.vast.config.vast_max_k=${VAST_MAX_K}"
  "algorithm.vast.config.vast_comp_coef=${VAST_COMP_COEF}"
  "algorithm.vast.config.vast_sampling_seed=${VAST_SAMPLING_SEED}"
  "algorithm.vast.config.expectile_tau=${EXPECTILE_TAU}"
  "algorithm.vast.config.g_lr=${G_LR}"
  "algorithm.vast.config.v_lr=${V_LR}"
  "algorithm.vast.config.output_reward_coef=${OUTPUT_REWARD_COEF}"
  "algorithm.vast.config.disc_reward_coef=${DISC_REWARD_COEF}"
  "algorithm.discriminator.inference.device=${DEVICE}"
  "algorithm.discriminator.learner_device=${DEVICE}"
  "algorithm.flow.device=${DEVICE}"
  "algorithm.flow.inference_device=${DEVICE}"
  "algorithm.trainer.batch_size=${POLICY_BATCH_SIZE}"
  "offline.episodes_path=${OFFLINE_EPISODES}"
  "offline.pretrain_data_path=${PRETRAIN_DATA}"
  "offline.num_train_steps=${NUM_TRAIN_STEPS}"
  "offline.vast_finetune.num_steps=${VAST_FINETUNE_STEPS}"
  "offline.vast_finetune.batch_size=${VAST_BATCH_SIZE}"
  "offline.branch_weight.beta=${BRANCH_BETA}"
  "offline.branch_weight.k=${BRANCH_K}"
  "offline.use_online_success=${USE_ONLINE_SUCCESS}"  # always: pure success rollouts -> pos_only
  "offline.run_subfix=${RUN_SUBFIX}"
  "offline.skip_rl=${SKIP_RL}"
)
if [[ "${SKIP_RL}" == "1" || "${SKIP_RL}" == "true" ]]; then
  if [[ -z "${VAST_FINETUNED}" ]]; then
    echo "[ERROR] SKIP_RL requires VAST_FINETUNED pointing to a schema-v7 or deprecated schema-v6 VAST checkpoint." >&2
    exit 1
  fi
  HYDRA_OVERRIDES+=("offline.vast_finetuned_path=${VAST_FINETUNED}")
fi
HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

echo "[train_offline_dipole] task=${TASK} device=${DEVICE} seed=${SEED} policy_steps=${NUM_TRAIN_STEPS} vast_finetune_steps=${VAST_FINETUNE_STEPS}"
echo "[train_offline_dipole] vast_batch=${VAST_BATCH_SIZE} policy_batch=${POLICY_BATCH_SIZE}"
echo "[train_offline_dipole] algorithm=vast_value_stitching_adaptation v_mode=${VAST_V_MODE} K=${VAST_MAX_K} comp_coef=${VAST_COMP_COEF} expectile_tau=${EXPECTILE_TAU} sampling_seed=${VAST_SAMPLING_SEED}"
echo "[train_offline_dipole] reward: output_coef=${OUTPUT_REWARD_COEF} disc_coef=${DISC_REWARD_COEF}"
echo "[train_offline_dipole] run_subfix=${RUN_SUBFIX} branch_beta=${BRANCH_BETA} branch_k=${BRANCH_K}"
echo "[train_offline_dipole] policy_ckpt=${POLICY_CKPT}"
echo "[train_offline_dipole] nnpu_ckpt=${NNPU_CKPT}"
echo "[train_offline_dipole] vast_ckpt=${VAST_CKPT}"
echo "[train_offline_dipole] skip_rl=${SKIP_RL} vast_finetuned=${VAST_FINETUNED}"
echo "[train_offline_dipole] episodes=${OFFLINE_EPISODES}"
echo "[train_offline_dipole] pretrain_data=${PRETRAIN_DATA}"
echo "[train_offline_dipole] use_online_success=${USE_ONLINE_SUCCESS}"

"${PY}" -m robosuite.pipeline.offline.src.train_offline_dipole "${HYDRA_OVERRIDES[@]}"
