set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=2

cd /home/gy/Documents/DAgger/robosuite/RL
python "${REPO_ROOT}/scripts/visualize.py" \
  --checkpoint /home/gy/Documents/DAgger/robosuite/RL/outputs/PandaLift_sac_lift_1_20260211_174800/actor_150000.pth \
  --dir /home/gy/Documents/DAgger/robosuite/RL/video \
  --num_episodes 50 \
  --img_size 256 \
  --fps 20