export CUDA_VISIBLE_DEVICE=0

python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/eval_flow.py \
  ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/flow/altogether/flow_policy_ep0060_20260306_145514.pt" \
  video_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/flow/altogether/video/ep60" \
  episodes=5 \
  succ_rate=true