set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=3

cd /home/gy/Documents/DAgger/robosuite/RL
python "${REPO_ROOT}/scripts/sac_drqv2_bc_lift_image.py" \
    --env-id "PandaLiftImage" \
    --pretrained-path "/home/gy/Documents/DAgger/robosuite/RL/models/resnet18-f37072fd.pth" \
    --wandb-mode "offline" \
    --capture-video \
    --total-timesteps 1000000 \
    --num_envs 5 \
    --image_size 128 \
    --video_freq 50 \
    --eval_freq 10000 \
    --batch_size 512 \
    --ent_scale 0.2 \
    --demo_data_path "/home/gy/Documents/DAgger/robosuite/RL/data/PandaLiftImage128" \
    --pretraining_steps 20000