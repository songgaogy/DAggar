#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate lpb

python -X faulthandler eval_base_diffusion_policy_pandalift.py \
  --policy_checkpoint '/home/dodo/Documents/DAggar/robosuite/lpb_robosuite/data/outputs/2026.03.06/13.42.42_train_diffusion_unet_hybrid_pandalift_image/checkpoints/100.ckpt' \
  --dataset_path '/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert' \
  --output_dir '/home/dodo/Documents/DAggar/robosuite/lpb_robosuite/data/release_pandalift_base_policy' \
  --device 'cuda:0' \
  --n_episodes 10 \
  --n_video 10 \
  --max_steps 300 \
  --seed_start 1000 \
  --camera_name 'agentview' \
  --render_obs_key 'agentview_image'
