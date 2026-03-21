export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl

# eval each task
python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/flow_multi/eval_flow.py \
  eval.ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/flow-20/flow_multi_ep0100_20260320_114720.pt" \
  eval.task_name='["PickPlaceBread","PickPlaceCereal","PickPlaceMilk","PickPlaceCan","Stack","Lift"]' \
  eval.video_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/multitask_6/flow-20/video/ep100" \
  eval.episodes=5 \
  eval.max_steps=500 \
  eval.succ_rate=false
