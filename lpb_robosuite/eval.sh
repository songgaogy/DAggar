#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate lpb

python -X faulthandler eval_test_time_optimization.py \
  --config-name=eval_pandalift \
  policy_checkpoint='/home/dodo/Documents/DAggar/robosuite/lpb_robosuite/data/outputs/2026.03.06/13.42.42_train_diffusion_unet_hybrid_pandalift_image/checkpoints' \
  dynamics_model_checkpoint='/home/dodo/Documents/DAggar/robosuite/lpb_robosuite/data/outputs/2026.03.06/15.28.37_pandalift/checkpoints' \
  demo_dataset_path='/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert' \
  dataset_path='/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert' \
  output_dir='/home/dodo/Documents/DAggar/robosuite/lpb_robosuite/data/release_pandalift_test'
