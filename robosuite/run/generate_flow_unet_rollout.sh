export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl

python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/flow_unet/generate_rollout_data.py \
  generate.ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/PickPlaceCan/flow_unet-10/flow_unet_ep0400_20260318_020356.pt" \
  generate.output_dir="/home/dodo/Documents/DAggar/robosuite/data/PandaPickPlaceCan/fail_rollout" \
  generate.device="cuda" \
  generate.episodes=100 \
  generate.keep_mode="fail" \
  generate.max_steps=400 \
  generate.n_ode_steps=20 \
  generate.action_horizon=4 \
  generate.render_height=256 \
  generate.render_width=256 \
  data.image_size=128
