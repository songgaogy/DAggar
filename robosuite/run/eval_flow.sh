export CUDA_VISIBLE_DEVICE=1

python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/eval_flow.py \
  ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/flow/BC_warmup/flow_policy_ep0020_20260301_231236.pt" \
  video_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/flow/BC_warmup/video/ep20" \
  episodes=5