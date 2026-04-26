#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"
PYTHON_BIN="${PYTHON_BIN:-/home/dodo/miniconda3/envs/daggar/bin/python}"

# ---------------------------------------------------------
ENVIRONMENT="${ENVIRONMENT:-Stack}"
DEMO_TASK_NAME="${DEMO_TASK_NAME:-PandaStack}"
TRAIN_EPISODE_MAX_STEPS="${TRAIN_EPISODE_MAX_STEPS:-500}"

EXPERT_NUM_TRAJ="${EXPERT_NUM_TRAJ:-20}"
SUCCESS_NUM_TRAJ="${SUCCESS_NUM_TRAJ:-80}"
FAIL_NUM_TRAJ="${FAIL_NUM_TRAJ:-80}"
NUM_TRAIN_STEP="${NUM_TRAIN_STEP:-200}"
VALUE_WARMUP_STEPS="${VALUE_WARMUP_STEPS:-20000}"
AWR_BETA="${AWR_BETA:-3.0}"
EXPECTILE="${EXPECTILE:-0.7}"
SUCCESS_REWARD_SCALE="${SUCCESS_REWARD_SCALE:-1.0}"
DISCRIMINATOR_REWARD_ENABLED="false"
DISCRIMINATOR_REWARD_SCALE="${DISCRIMINATOR_REWARD_SCALE:-1.0}"
DISCRIMINATOR_REWARD_CLIP="${DISCRIMINATOR_REWARD_CLIP:-5.0}"
QV_CACHE_ENABLED="${QV_CACHE_ENABLED:-true}"
QV_CACHE_FORCE_REBUILD="${QV_CACHE_FORCE_REBUILD:-false}"
# ---------------------------------------------------------

INTERACTIVE="${INTERACTIVE:-true}"
VIEWER_ENABLED="${VIEWER_ENABLED:-true}"
LEARNER_DEVICE="${LEARNER_DEVICE:-cuda:0}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cuda:1}"
DISCRIMINATOR_DEVICE="${DISCRIMINATOR_DEVICE:-cuda:1}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-4}"
N_ODE_STEPS="${N_ODE_STEPS:-8}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}"
LPB_CHECKPOINT="${LPB_CHECKPOINT:-/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/lpb_new/lpb_new_20260324_232921/lpb_new_20260324_232921_ep0050.pt}"
FPS_LOG_INTERVAL="${FPS_LOG_INTERVAL:-3.0}"
UNTHROTTLED="${UNTHROTTLED:-false}"

ONLINE_UPDATES="${ONLINE_UPDATES:-true}"
INTERVENTION_ENABLED="${INTERVENTION_ENABLED:-true}"
ASYNC_UPDATES="${ASYNC_UPDATES:-true}"
VISUALIZE_GRIPPER_MARKERS="${VISUALIZE_GRIPPER_MARKERS:-true}"
VIEWER_ASYNC="${VIEWER_ASYNC:-true}"
VIEWER_BACKEND="${VIEWER_BACKEND:-mjviewer}"
RENDER_FPS="${RENDER_FPS:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
BUFFER_SAVE_INTERVAL="${BUFFER_SAVE_INTERVAL:-5000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
PUBLISH_INTERVAL="${PUBLISH_INTERVAL:-300}"
SEED_VALUE="${SEED:-42}"
EPISODE_PAUSE_SEC="${EPISODE_PAUSE_SEC:-0}"

LOAD="${LOAD:-null}"
RESUME="${RESUME:-false}"
CHECKPOINT="${CHECKPOINT:-null}"
EXTRA_ARGS=("$@")

export ROOT_DIR
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_ENTITY="${WANDB_ENTITY:-songgao-personal}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

if [[ "${INTERACTIVE}" == "true" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

if [[ "${INTERACTIVE}" == "true" && -z "${DISPLAY:-}" ]]; then
  echo "[ERROR] INTERACTIVE=true requires a desktop X session, but DISPLAY is empty." >&2
  exit 1
fi

if [[ "${INTERVENTION_ENABLED}" == "true" && -z "${DISPLAY:-}" ]]; then
  echo "[ERROR] intervention.enabled=true currently requires DISPLAY because robosuite input devices import pynput." >&2
  exit 1
fi

cd "${ROOT_DIR}"

if [[ -n "${LOAD}" && "${LOAD}" != "null" ]]; then
  if [[ -d "${LOAD}" ]]; then
    if [[ -f "${LOAD}/checkpoints/latest.pt" ]]; then
      CHECKPOINT="${LOAD}/checkpoints/latest.pt"
    elif [[ -f "${LOAD}/latest.pt" ]]; then
      CHECKPOINT="${LOAD}/latest.pt"
    else
      echo "[ERROR] LOAD points to a directory, but no latest checkpoint was found under ${LOAD}" >&2
      exit 1
    fi
  else
    CHECKPOINT="${LOAD}"
  fi
  RESUME=true
fi

if [[ "${CHECKPOINT}" != "null" && ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Checkpoint file does not exist: ${CHECKPOINT}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m robosuite.pipeline.train_awr \
  seed="${SEED_VALUE}" \
  env.environment="${ENVIRONMENT}" \
  env.renderer="mjviewer" \
  data.task_name="${DEMO_TASK_NAME}" \
  data.expert_num_trajectories="${EXPERT_NUM_TRAJ}" \
  data.success_num_trajectories="${SUCCESS_NUM_TRAJ}" \
  data.fail_num_trajectories="${FAIL_NUM_TRAJ}" \
  runtime.interactive="${INTERACTIVE}" \
  runtime.viewer_enabled="${VIEWER_ENABLED}" \
  runtime.visualize_gripper_markers="${VISUALIZE_GRIPPER_MARKERS}" \
  runtime.render_fps="${RENDER_FPS}" \
  runtime.viewer_async="${VIEWER_ASYNC}" \
  runtime.viewer_backend="${VIEWER_BACKEND}" \
  runtime.fps_log_interval="${FPS_LOG_INTERVAL}" \
  runtime.buffer_save_interval="${BUFFER_SAVE_INTERVAL}" \
  runtime.qv_cache.enabled="${QV_CACHE_ENABLED}" \
  runtime.qv_cache.force_rebuild="${QV_CACHE_FORCE_REBUILD}" \
  runtime.discriminator_reward_enabled="${DISCRIMINATOR_REWARD_ENABLED}" \
  runtime.unthrottled="${UNTHROTTLED}" \
  runtime.async_updates="${ASYNC_UPDATES}" \
  runtime.online_updates_enabled="${ONLINE_UPDATES}" \
  runtime.episode_pause_sec="${EPISODE_PAUSE_SEC}" \
  runtime.train_episode_max_steps="${TRAIN_EPISODE_MAX_STEPS}" \
  runtime.num_train_step="${NUM_TRAIN_STEP}" \
  intervention.enabled="${INTERVENTION_ENABLED}" \
  runtime.resume="${RESUME}" \
  runtime.checkpoint="${CHECKPOINT}" \
  runtime.init_checkpoint="${INIT_CHECKPOINT}" \
  discriminator.enabled="${DISCRIMINATOR_REWARD_ENABLED}" \
  discriminator.model.lpb_ckpt="${LPB_CHECKPOINT}" \
  discriminator.policy.ckpt="${INIT_CHECKPOINT}" \
  discriminator.policy.device="${DISCRIMINATOR_DEVICE}" \
  discriminator.feature.device="${DISCRIMINATOR_DEVICE}" \
  discriminator.detector.device="${DISCRIMINATOR_DEVICE}" \
  algorithm.trainer.value_warmup_steps="${VALUE_WARMUP_STEPS}" \
  algorithm.awr.device="${LEARNER_DEVICE}" \
  algorithm.awr.inference_device="${INFERENCE_DEVICE}" \
  algorithm.awr.action_horizon="${ACTION_HORIZON}" \
  algorithm.awr.execute_horizon="${EXECUTE_HORIZON}" \
  algorithm.awr.n_ode_steps="${N_ODE_STEPS}" \
  algorithm.awr.beta="${AWR_BETA}" \
  algorithm.awr.expectile="${EXPECTILE}" \
  algorithm.awr.success_reward_scale="${SUCCESS_REWARD_SCALE}" \
  algorithm.awr.discriminator_reward_scale="${DISCRIMINATOR_REWARD_SCALE}" \
  algorithm.awr.discriminator_reward_clip="${DISCRIMINATOR_REWARD_CLIP}" \
  algorithm.trainer.batch_size="${TRAIN_BATCH_SIZE}" \
  algorithm.trainer.steps_per_update="${PUBLISH_INTERVAL}" \
  logging.log_interval="${LOG_INTERVAL}" \
  logging.checkpoint_interval="${CHECKPOINT_INTERVAL}" \
  "${EXTRA_ARGS[@]}"

# Examples:
#   bash robosuite/pipeline/scripts/train_awr.sh
#   ENVIRONMENT=PickPlaceCan DEMO_TASK_NAME=PandaPickPlaceCan bash robosuite/pipeline/scripts/train_awr.sh
#   INTERACTIVE=false VIEWER_ENABLED=false INTERVENTION_ENABLED=false bash robosuite/pipeline/scripts/train_awr.sh
#   DISCRIMINATOR_REWARD_ENABLED=false bash robosuite/pipeline/scripts/train_awr.sh
#   LOAD=outputs/awr/awr_Stack_YYYY-mm-dd_HH-MM-SS bash robosuite/pipeline/scripts/train_awr.sh
