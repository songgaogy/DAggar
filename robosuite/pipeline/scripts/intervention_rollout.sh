export CUDA_VISIBLE_DEVICES=0

cd ~/Documents/DAggar/robosuite
python -m robosuite.pipeline.collect_human_intervention \
  policy.type=flow_multi \
  discriminator.type=lpb_dice \
  intervention.environment=Lift
