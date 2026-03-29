export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

# eval each task
# python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/flow_multi/eval_flow.py \
#   eval.ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt" \
#   eval.task_name='["PickPlaceBread","PickPlaceCereal","PickPlaceMilk","PickPlaceCan","Stack","Lift"]' \
#   eval.video_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/policy/flow-20/video/ep100" \
#   eval.episodes=0 \
#   eval.max_steps=500 \
#   eval.succ_rate=true

python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/flow_multi/eval_flow.py \
  eval.ckpt="/home/dodo/Documents/DAggar/robosuite/outputs/flow-DAgger/flow_dagger_PickPlaceBread_2026-03-30_01-54-07/checkpoints/step_00010000_updates_00009645_ep_00034.pt" \
  eval.task_name='["PickPlaceBread"]' \
  eval.video_dir="/home/dodo/Documents/DAggar/robosuite/outputs/flow-DAgger/flow_dagger_PickPlaceBread_2026-03-30_01-54-07/checkpoints/step100/video" \
  eval.episodes=5 \
  eval.max_steps=500 \
  eval.succ_rate=true
