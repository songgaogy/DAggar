set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite_dipole-rl_v3}"
cd "$ROOT_DIR"
PY="${PY:-$HOME/miniconda3/envs/dagger/bin/python}"
export CUDA_VISIBLE_DEVICES=1

# -------------------------------------
TASK="PickPlaceCereal"
POLICY_CKPT="checkpoints/multitask_6/flow_multi_ep0100.pt"
NNPU_CKPT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_194127_PickPlaceCereal/checkpoints/pu_bce_head.pth"
IQL_CKPT="outputs/dipole_rl-iql/offline_iql_qv-v2_explore-ensemble/PickPlaceCereal/iql_state.pt"   # Pretrained IQL to CONTINUE training from (init_iql_qv.sh output).
OFFLINE_EPISODES="data/${TASK}/offline_data/offline_episodes.pt"  # Collected offline episodes (collect_data.sh output).
PRETRAIN_DATA="data/${TASK}/pretrain_data-20260615_174814"
DEVICE="${DEVICE:-cuda:0}"
NUM_TRAIN_STEPS=15000
V_FINETUNE_STEPS=10000
V_BATCH_SIZE=512
POLICY_BATCH_SIZE=256

# w_pos = sigmoid(beta * (G + k)); decision boundary at G=-k.
BRANCH_BETA=1
BRANCH_K=-2
RUN_SUBFIX="beta${BRANCH_BETA}_k${BRANCH_K}_ensemble-GAE"

# SKIP_RL=1: skip Phase A IQL finetune; reuse IQL_FINETUNED and copy it into the
# new run dir as checkpoints/iql_state_finetuned.pt (see train_offline_dipole.py).
SKIP_RL=1
IQL_FINETUNED="outputs/dipole-rl-offline/PickPlaceCereal_20260713_113548_beta1_k-2_ensemble-GAE/checkpoints/iql_state_finetuned.pt"
USE_ONLINE_SUCCESS=0
# -------------------------------------

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HYDRA_FULL_ERROR=1
# shellcheck disable=SC1091
source "${ROOT_DIR}/robosuite/pipeline/scripts/utils/hydra_disable_outputs.sh"

HYDRA_OVERRIDES=(
  "env.environment=${TASK}"
  "runtime.init_checkpoint=${POLICY_CKPT}"
  "algorithm.discriminator.checkpoint=${NNPU_CKPT}"
  "algorithm.q_learning.warmup_ckpt=${IQL_CKPT}"
  "algorithm.q_learning.config.device=${DEVICE}"
  "algorithm.discriminator.inference.device=${DEVICE}"
  "algorithm.discriminator.learner_device=${DEVICE}"
  "algorithm.flow.device=${DEVICE}"
  "algorithm.flow.inference_device=${DEVICE}"
  "algorithm.trainer.batch_size=${POLICY_BATCH_SIZE}"
  "offline.episodes_path=${OFFLINE_EPISODES}"
  "offline.pretrain_data_path=${PRETRAIN_DATA}"
  "offline.num_train_steps=${NUM_TRAIN_STEPS}"
  "offline.iql_finetune.num_steps=${V_FINETUNE_STEPS}"
  "offline.iql_finetune.batch_size=${V_BATCH_SIZE}"
  "offline.branch_weight.beta=${BRANCH_BETA}"
  "offline.branch_weight.k=${BRANCH_K}"
  "offline.use_online_success=${USE_ONLINE_SUCCESS}"  # always: pure success rollouts -> pos_only
  "offline.run_subfix=${RUN_SUBFIX}"
  "offline.skip_rl=${SKIP_RL}"
)
if [[ "${SKIP_RL}" == "1" || "${SKIP_RL}" == "true" ]]; then
  HYDRA_OVERRIDES+=("offline.iql_finetuned_path=${IQL_FINETUNED}")
fi
HYDRA_OVERRIDES+=("${HYDRA_DISABLE_LOG_OVERRIDES[@]}")

echo "[train_offline_dipole] task=${TASK} device=${DEVICE} policy_steps=${NUM_TRAIN_STEPS} v_steps=${V_FINETUNE_STEPS}"
echo "[train_offline_dipole] v_batch=${V_BATCH_SIZE} policy_batch=${POLICY_BATCH_SIZE}"
echo "[train_offline_dipole] run_subfix=${RUN_SUBFIX} branch_beta=${BRANCH_BETA} branch_k=${BRANCH_K}"
echo "[train_offline_dipole] policy_ckpt=${POLICY_CKPT}"
echo "[train_offline_dipole] nnpu_ckpt=${NNPU_CKPT}"
echo "[train_offline_dipole] iql_ckpt=${IQL_CKPT}"
echo "[train_offline_dipole] skip_rl=${SKIP_RL} iql_finetuned=${IQL_FINETUNED}"
echo "[train_offline_dipole] episodes=${OFFLINE_EPISODES}"
echo "[train_offline_dipole] pretrain_data=${PRETRAIN_DATA}"
echo "[train_offline_dipole] use_online_success=${USE_ONLINE_SUCCESS}"

"${PY}" -m robosuite.pipeline.offline.src.train_offline_dipole "${HYDRA_OVERRIDES[@]}"
