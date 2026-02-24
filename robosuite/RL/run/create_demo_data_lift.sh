set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=7

# 15, 47, 96
cd /home/gy/Documents/DAgger/robosuite/RL
python "${REPO_ROOT}/scripts/create_demo_data_lift.py" \
    --checkpoint /home/gy/Documents/DAgger/robosuite/RL/outputs/PandaLift_sac_lift_1_20260211_174800/actor_960000.pth \
    --output /home/gy/Documents/DAgger/robosuite/RL/data/PandaLiftImage128/success/step-960k/sac_lift_128_960000.hdf5 \
    --num_episodes 1000 \
    --img_size 128 \
    --type success \
    --max_episode_steps 60
