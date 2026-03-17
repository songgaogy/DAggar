export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="offline"
export WANDB_ENTITY="songgao-personal"

# --- for 350 trajectroies ---
# python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/flow_unet/train_flow.py \
#   data.data_dir="/home/dodo/Documents/DAggar/robosuite/data/PandaPickPlaceCan/expert_recover" \
#   data.camera_names='["agentview","robot0_robotview","robot0_eye_in_hand"]' \
#   checkpoint.save_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/PickPlaceCan/flow_unet-200" \
#   data.preload_all_demos_to_ram=true \
#   data.use_disk_cache=true \
#   data.num_train_traj=100 \
#   train.batch_size=512 \
#   train.num_workers=16 \
#   train.prefetch_factor=8 \
#   train.device="cuda"


python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/flow_unet/train_flow.py \
  data.data_dir="/home/dodo/Documents/DAggar/robosuite/data/PandaPickPlaceCan/expert_recover" \
  data.camera_names='["agentview","robot0_robotview","robot0_eye_in_hand"]' \
  checkpoint.save_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/PickPlaceCan/flow_unet-10" \
  data.preload_all_demos_to_ram=true \
  data.use_disk_cache=true \
  data.num_train_traj=10 \
  train.epochs=400 \
  train.batch_size=256 \
  train.num_workers=16 \
  train.prefetch_factor=8 \
  flow.head.hidden_dim=128 \
  flow.head.time_dim=128 \
  flow.fusion.num_heads=4 \
  flow.head.cond_dim=256 \
  checkpoint.save_freq=100 \
  train.device="cuda"
