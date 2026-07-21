set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=1

# -------------------------------------
INPUT="outputs/dipole-rl-offline_vast-disc_reward/PickPlaceCereal_20260719_212234"
TASK="PickPlaceCereal"
POLICY_CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"
VAST_CKPT="outputs/dipole_rl-vast/tau0p7_en5_disc-reward0p1/PickPlaceCereal/vast_state.pt"
VAST_V_MODE="indep_ensemble"
EXPECTILE_TAU=0.7

ADVANTAGE_ESTIMATOR="td1"
GAE_LAMBDA=0.6
BRANCH_BETA=2
BRANCH_K=0   # w_pos = sigmoid(beta * (G + k)); decision boundary at G=-k.
RUN_SUBFIX="vast-${VAST_V_MODE}_beta${BRANCH_BETA}_k${BRANCH_K}_tau0p7"

USE_ONLINE_SUCCESS=0

NUM_TRAIN_STEPS=15000
VAST_FINETUNE_STEPS=20000
# -------------------------------------

OFFLINE_EPISODES="${OFFLINE_EPISODES:-data/${TASK}/offline_data/offline_episodes.pt}"
PRETRAIN_DATA="${PRETRAIN_DATA:-data/${TASK}/pretrain_data-20260615_174814}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
VAST_BATCH_SIZE="${VAST_BATCH_SIZE:-512}"
POLICY_BATCH_SIZE="${POLICY_BATCH_SIZE:-256}"
SKIP_RL="${SKIP_RL:-true}"

PIPELINE_RUN_INPUT="${1:-${INPUT}}"
if [[ ! -d "${PIPELINE_RUN_INPUT}" ]]; then
  echo "[ERROR] Pipeline run directory does not exist: ${PIPELINE_RUN_INPUT}" >&2
  exit 1
fi
PIPELINE_RUN_DIR="$(realpath "${PIPELINE_RUN_INPUT}")"
RUN_INFO="${PIPELINE_RUN_DIR}/discriminator/run_info.json"
DISCRIMINATOR_CKPT="${PIPELINE_RUN_DIR}/discriminator/checkpoints/pu_bce_head_finetuned.pth"
if [[ "${SKIP_RL}" == "1" || "${SKIP_RL}" == "true" ]]; then
  SKIP_RL="true"
  STAGE_NAME="${STAGE_NAME:-dipole_skip_rl}"
else
  SKIP_RL="false"
  STAGE_NAME="${STAGE_NAME:-dipole}"
fi
STAGE_RUN_DIR="${PIPELINE_RUN_DIR}/${STAGE_NAME}"
VAST_FINETUNED_CKPT="${VAST_FINETUNED_CKPT:-${PIPELINE_RUN_DIR}/dipole/checkpoints/vast_state_finetuned.pt}"
if [[ ! -f "${RUN_INFO}" ]]; then
  echo "[ERROR] Discriminator run metadata does not exist: ${RUN_INFO}" >&2
  exit 1
fi
if [[ ! -f "${DISCRIMINATOR_CKPT}" ]]; then
  echo "[ERROR] Finetuned discriminator checkpoint does not exist: ${DISCRIMINATOR_CKPT}" >&2
  exit 1
fi
RUN_TASK="$("${PY}" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["task_name"])' "${RUN_INFO}")"
if [[ "${RUN_TASK}" != "${TASK}" ]]; then
  echo "[ERROR] Run task mismatch: metadata=${RUN_TASK} launcher=${TASK}." >&2
  exit 1
fi


export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1
# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

HYDRA_OVERRIDES=(
  "seed=${SEED}"
  "env.environment=${TASK}"
  "runtime.init_checkpoint=${POLICY_CKPT}"
  "algorithm.discriminator.checkpoint=${DISCRIMINATOR_CKPT}"
  "algorithm.vast.warmup_ckpt=${VAST_CKPT}"
  "algorithm.vast.config.device=${DEVICE}"
  "algorithm.vast.config.vast_v_mode=${VAST_V_MODE}"
  "algorithm.vast.config.expectile_tau=${EXPECTILE_TAU}"
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
  "offline.vast_finetune.relabel_disc_reward=true"
  "offline.advantage.estimator=${ADVANTAGE_ESTIMATOR}"
  "offline.advantage.gae_lambda=${GAE_LAMBDA}"
  "offline.branch_weight.beta=${BRANCH_BETA}"
  "offline.branch_weight.k=${BRANCH_K}"
  "offline.use_online_success=${USE_ONLINE_SUCCESS}"  # always: pure success rollouts -> pos_only
  "offline.run_subfix=${RUN_SUBFIX}"
  "offline.run_dir=${STAGE_RUN_DIR}"
  "offline.skip_rl=${SKIP_RL}"
)
if [[ "${SKIP_RL}" == "true" ]]; then
  HYDRA_OVERRIDES+=("offline.vast_finetuned_path=${VAST_FINETUNED_CKPT}")
fi
HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

echo "[train_offline_dipole] task=${TASK} device=${DEVICE} seed=${SEED} policy_steps=${NUM_TRAIN_STEPS} vast_finetune_steps=${VAST_FINETUNE_STEPS}"
echo "[train_offline_dipole] pipeline_run_dir=${PIPELINE_RUN_DIR}"
echo "[train_offline_dipole] stage_run_dir=${STAGE_RUN_DIR}"
echo "[train_offline_dipole] vast_batch=${VAST_BATCH_SIZE} policy_batch=${POLICY_BATCH_SIZE}"
echo "[train_offline_dipole] algorithm=vast_value_stitching_adaptation v_mode=${VAST_V_MODE} expectile_tau=${EXPECTILE_TAU}"
echo "[train_offline_dipole] advantage=${ADVANTAGE_ESTIMATOR} gae_lambda=${GAE_LAMBDA}"
echo "[train_offline_dipole] run_subfix=${RUN_SUBFIX} branch_beta=${BRANCH_BETA} branch_k=${BRANCH_K}"
echo "[train_offline_dipole] policy_ckpt=${POLICY_CKPT}"
echo "[train_offline_dipole] discriminator_ckpt=${DISCRIMINATOR_CKPT}"
echo "[train_offline_dipole] vast_ckpt=${VAST_CKPT}"
echo "[train_offline_dipole] skip_rl=${SKIP_RL} relabel_disc_reward=true"
if [[ "${SKIP_RL}" == "true" ]]; then
  echo "[train_offline_dipole] vast_finetuned_ckpt=${VAST_FINETUNED_CKPT}"
fi
echo "[train_offline_dipole] episodes=${OFFLINE_EPISODES}"
echo "[train_offline_dipole] pretrain_data=${PRETRAIN_DATA}"
echo "[train_offline_dipole] use_online_success=${USE_ONLINE_SUCCESS}"

"${PY}" -m robosuite.pipeline.offline.src.train_offline_dipole "${HYDRA_OVERRIDES[@]}"
