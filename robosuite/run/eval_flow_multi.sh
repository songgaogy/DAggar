export CUDA_VISIBLE_DEVICES=0

python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/flow_multi/eval_flow.py \
  eval.ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/flow_multi/multitask/flow_multi_ep0010.pt" \
  eval.task_name="PickPlaceCan" \
  eval.video_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/flow_multi/multitask/videos/pickplace" \
  eval.episodes=5 \
  eval.succ_rate=true
