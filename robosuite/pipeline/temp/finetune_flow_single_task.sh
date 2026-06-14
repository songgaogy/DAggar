#!/usr/bin/env bash
set -euo pipefail

cd /home/dodo/Documents/DAggar/robosuite

PYTHON="${PYTHON:-/home/dodo/miniconda3/envs/dagger/bin/python}"
TASK="PickPlaceBread"
TRAIN_DEVICE="cuda:0"
NUM_TRAJECTORIES="20"
STEPS="2000"
SAVE_INTERVAL="500"

export WANDB_NAME="${WANDB_NAME:-songgao-personal}"
export WANDB_MODE="disabled"

RUN_SPECS=(
  "flow_head 3e-6 ft_flowhead_lr3e-6_ema_bnfreeze"
  "flow_head 1e-5 ft_flowhead_lr1e-5_ema_bnfreeze"
  "flow_head_aggregator 3e-6 ft_flowhead_aggregator_lr3e-6_ema_bnfreeze"
)

for spec in "${RUN_SPECS[@]}"; do
  read -r TRAIN_SCOPE LEARNING_RATE OUTPUT_POSTFIX <<<"${spec}"
  "${PYTHON}" -m robosuite.pipeline.temp.train_flow_offline \
    --env "${TASK}" \
    --num-trajectories "${NUM_TRAJECTORIES}" \
    --steps "${STEPS}" \
    --save-interval "${SAVE_INTERVAL}" \
    --learning-rate "${LEARNING_RATE}" \
    --train-scope "${TRAIN_SCOPE}" \
    --ema-decay 0.999 \
    --freeze-visual-bn \
    --device "${TRAIN_DEVICE}" \
    --output-root "./outputs/flow_dagger_no-hil" \
    --output-postfix "${OUTPUT_POSTFIX}"
done
