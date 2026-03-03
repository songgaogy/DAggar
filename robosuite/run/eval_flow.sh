export CUDA_VISIBLE_DEVICE=0

python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/eval_flow.py \
  ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/flow/BC_warmup/flow_policy_ep0040_20260303_135632.pt" \
  video_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/flow/BC_warmup/video/ep40" \
  episodes=5 \
  succ_rate=true