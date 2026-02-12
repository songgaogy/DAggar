set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1

cd /home/gy/Documents/DAgger/robosuite/RL
python "${REPO_ROOT}/scripts/sac_lift.py" \
    --env-id "PandaLift" \
    --wandb-mode "offline" \
    --capture-video \
    --total-timesteps 1000000 \
    --num_envs 5 \
    --buffer_size 100000 \
    --video_freq 50 \
    --eval_freq 10000 \
    --batch_size 1024 \
    --success_terminate