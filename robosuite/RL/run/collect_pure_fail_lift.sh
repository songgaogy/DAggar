set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=7

# cd /home/gy/Documents/DAgger/robosuite/RL
# python "${REPO_ROOT}/scripts/collect_pure_fail_lift.py" \
#     --checkpoint /home/gy/Documents/DAgger/robosuite/RL/outputs/PandaLift_sac_lift_1_20260211_174800/actor_960000.pth \
#     --output /home/gy/Documents/DAgger/robosuite/RL/data/PandaLiftImage128/pure_fail/step-960k/sac_lift_128_960000_fail.hdf5 \
#     --num_episodes 2000 \
#     --img_size 128 \
#     --max_episode_steps 60

# cd /home/gy/Documents/DAgger/robosuite/RL
# python "${REPO_ROOT}/scripts/collect_pure_fail_lift.py" \
#     --checkpoint /home/gy/Documents/DAgger/robosuite/RL/outputs/PandaLift_sac_lift_1_20260211_174800/actor_150000.pth \
#     --output /home/gy/Documents/DAgger/robosuite/RL/data/PandaLiftImage128/pure_fail/step-150k/sac_lift_128_150000_fail.hdf5 \
#     --num_episodes 2000 \
#     --img_size 128 \
#     --max_episode_steps 60

cd /home/gy/Documents/DAgger/robosuite/RL
python "${REPO_ROOT}/scripts/collect_pure_fail_lift.py" \
    --checkpoint /home/gy/Documents/DAgger/robosuite/RL/outputs/PandaLift_sac_lift_1_20260211_174800/actor_470000.pth \
    --output /home/gy/Documents/DAgger/robosuite/RL/data/PandaLiftImage128/pure_fail/step-470k/sac_lift_128_470000_fail.hdf5 \
    --num_episodes 2000 \
    --img_size 128 \
    --max_episode_steps 60