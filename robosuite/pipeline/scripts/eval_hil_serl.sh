export CUDA_VISIBLE_DEVICES=0

cd ~/Documents/DAggar/robosuite
/home/dodo/miniconda3/envs/daggar/bin/python -m robosuite.pipeline.train_hil_serl \
  runtime.max_steps=2000 \
  runtime.interactive=false \
  runtime.eval_deterministic=true \
  intervention.enabled=false \
  runtime.checkpoint=./outputs/hil_serl/YOUR_RUN/checkpoints/latest.pt
