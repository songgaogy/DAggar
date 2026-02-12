set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1

cd /home/gy/Documents/DAgger/robosuite/RL
python "${REPO_ROOT}/scripts/create_demo_data_lift.py" \
    --checkpoint /home/gy/Documents/DAgger/robosuite/RL/outputs/PandaLift_sac_lift_1_20260211_174800/actor_470000.pth \
    --output /home/gy/Documents/DAgger/robosuite/RL/data/PandaLift/sac_lift_128_470000.hdf5 \
    --num_episodes 100 \
    --img_size 128
