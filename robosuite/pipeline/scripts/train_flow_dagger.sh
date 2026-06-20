#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/Documents/DAggar/robosuite}"

# ----------------------------------------------------------------------
ENVIRONMENT="NutAssemblySquare"
NUM_TRAJECTORIES=30   # pretraining data
UPDATE_PER_STEP=1     # 1 env step -> x policy updates
SAVE_FREQ="${SAVE_FREQ:-5000}"        # save checkpoint every x env steps
DEMO_SPLIT="pretrain_data-20260615_174814"
SEED_VALUE=42
RUN_NAME_SUFFIX="_${NUM_TRAJECTORIES}pretrain_seed${SEED_VALUE}"

INIT_CHECKPOINT="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/flow_multi_ep0100.pt"
DISCRIMINATOR_CHECKPOINT="checkpoints/dyn_disc/pu_bce_eval_robosuite/run_20260619_203140_NutAssemblySquare/checkpoints/pu_bce_head.pth"
LEARNER_DEVICE="cuda:1"
INFERENCE_DEVICE="cuda:0"
ACTION_HORIZON=8
EXECUTE_HORIZON=8
# ----------------------------------------------------------------------

DEMO_TASK_NAME="${ENVIRONMENT}"
DISCRIMINATOR_TASK_NAME="${ENVIRONMENT}"
INTERACTIVE="true"
export MUJOCO_GL="${MUJOCO_GL:-glfw}"
VIEWER_ENABLED="${VIEWER_ENABLED:-true}"
IMAGE_OBS_FPS="${IMAGE_OBS_FPS:-20}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-0}"

# Async learner backpressure: cap queued updates so the learner cannot fall far behind the
# rollout (avoids stale-progress backlog + end-of-run flush burst). 0 = unbounded (legacy).
MAX_PENDING_UPDATES="${MAX_PENDING_UPDATES:-8}"
N_ODE_STEPS=10
FPS_LOG_INTERVAL="${FPS_LOG_INTERVAL:-3.0}"
UNTHROTTLED="${UNTHROTTLED:-false}"

ONLINE_UPDATES="${ONLINE_UPDATES:-true}"
INTERVENTION_ENABLED="${INTERVENTION_ENABLED:-true}"
ASYNC_UPDATES="${ASYNC_UPDATES:-true}"
VISUALIZE_GRIPPER_MARKERS="${VISUALIZE_GRIPPER_MARKERS:-true}"
LOGGING_USE_WANDB="${LOGGING_USE_WANDB:-false}"
VIEWER_ASYNC="${VIEWER_ASYNC:-true}"
VIEWER_BACKEND="${VIEWER_BACKEND:-mjviewer}"
RENDER_FPS="${RENDER_FPS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
BUFFER_SAVE_INTERVAL="${BUFFER_SAVE_INTERVAL:-5000}"
CHECKPOINT_INTERVAL="${SAVE_FREQ}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
PUBLISH_INTERVAL="${PUBLISH_INTERVAL:-300}"
EPISODE_PAUSE_SEC="${EPISODE_PAUSE_SEC:-0}"

LOAD="${LOAD:-null}"
RESUME="${RESUME:-false}"
CHECKPOINT="${CHECKPOINT:-null}"
EXTRA_ARGS=("$@")

export ROOT_DIR
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

cd "${ROOT_DIR}"
python -m robosuite.pipeline.train_flow_dagger \
  seed="${SEED_VALUE}" \
  env.environment="${ENVIRONMENT}" \
  env.renderer="mjviewer" \
  data.task_name="${DEMO_TASK_NAME}" \
  data.num_trajectories="${NUM_TRAJECTORIES}" \
  runtime.interactive="${INTERACTIVE}" \
  runtime.viewer_enabled="${VIEWER_ENABLED}" \
  runtime.visualize_gripper_markers="${VISUALIZE_GRIPPER_MARKERS}" \
  runtime.image_obs_fps="${IMAGE_OBS_FPS}" \
  runtime.render_fps="${RENDER_FPS}" \
  runtime.viewer_async="${VIEWER_ASYNC}" \
  runtime.viewer_backend="${VIEWER_BACKEND}" \
  runtime.fps_log_interval="${FPS_LOG_INTERVAL}" \
  runtime.buffer_save_interval="${BUFFER_SAVE_INTERVAL}" \
  runtime.save_transition_chunks=false \
  runtime.save_demo_buffer=true \
  runtime.unthrottled="${UNTHROTTLED}" \
  runtime.async_updates="${ASYNC_UPDATES}" \
  runtime.online_updates_enabled="${ONLINE_UPDATES}" \
  runtime.episode_pause_sec="${EPISODE_PAUSE_SEC}" \
  runtime.load_buffers=false \
  intervention.enabled="${INTERVENTION_ENABLED}" \
  runtime.resume="${RESUME}" \
  runtime.checkpoint="${CHECKPOINT}" \
  runtime.init_checkpoint="${INIT_CHECKPOINT}" \
  algorithm.trainer.pretrain_steps="${PRETRAIN_STEPS}" \
  algorithm.flow.device="${LEARNER_DEVICE}" \
  algorithm.flow.inference_device="${INFERENCE_DEVICE}" \
  algorithm.flow.action_horizon="${ACTION_HORIZON}" \
  algorithm.flow.execute_horizon="${EXECUTE_HORIZON}" \
  algorithm.flow.n_ode_steps="${N_ODE_STEPS}" \
  algorithm.trainer.batch_size="${TRAIN_BATCH_SIZE}" \
  algorithm.trainer.update_per_step="${UPDATE_PER_STEP}" \
  algorithm.trainer.steps_per_update="${PUBLISH_INTERVAL}" \
  algorithm.trainer.max_pending_updates="${MAX_PENDING_UPDATES}" \
  data.demo_split="${DEMO_SPLIT}" \
  logging.use_wandb="${LOGGING_USE_WANDB}" \
  logging.run_name_suffix="${RUN_NAME_SUFFIX}" \
  logging.log_interval="${LOG_INTERVAL}" \
  logging.checkpoint_interval="${CHECKPOINT_INTERVAL}" \
  discriminator.task_name="${DISCRIMINATOR_TASK_NAME}" \
  discriminator.checkpoint="${DISCRIMINATOR_CHECKPOINT}" \
  "${EXTRA_ARGS[@]}"
