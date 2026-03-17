export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl


# --- for 350 trajectroies ---
# python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/flow_unet/eval_flow.py \
#   eval.ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/PickPlaceCan/flow_unet-200/flow_unet_ep0040_20260318_005520.pt" \
#   eval.video_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/PickPlaceCan/flow_unet-200/videos/ep40" \
#   eval.device="cuda" \
#   eval.episodes=5 \
#   eval.succ_rate=false \
#   eval.max_steps=400 \
#   eval.n_ode_steps=20 \
#   eval.action_horizon=4 \
#   eval.video_camera="agentview" \
#   data.image_size=128


python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/flow_unet/eval_flow.py \
  eval.ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/PickPlaceCan/flow_unet-10/flow_unet_ep0400_20260318_020356.pt" \
  eval.video_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/PickPlaceCan/flow_unet-10/videos/ep400" \
  eval.device="cuda" \
  eval.episodes=10 \
  eval.succ_rate=false \
  eval.max_steps=400 \
  eval.n_ode_steps=20 \
  eval.action_horizon=4 \
  flow.fusion.num_heads=4 \
  flow.head.cond_dim=256 \
  flow.head.hidden_dim=128 \
  flow.head.time_dim=128 \
  eval.video_camera="agentview" \
  data.image_size=128