set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1

cd /home/dodo/Documents/DAggar/robosuite/robosuite/RL
python "${REPO_ROOT}/scripts/sac_continuous_action.py" \
    --env-id "PandaLiftImage" \
    --pretrained-path "/home/dodo/Documents/DAggar/robosuite/robosuite/RL/models/resnet18-f37072fd.pth" \
    --wandb-mode "offline" \
    --capture-video \
    --total-timesteps 1000000 \
    --num_envs 5 \
    --buffer_size 50000 \
    --image_size 84 \
    --video_freq 10 \
    --eval_freq 10000