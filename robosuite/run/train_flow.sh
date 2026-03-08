export CUDA_VISIBLE_DEVICES=0

python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/train_flow.py \
  data_dirs='["/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert","/home/dodo/Documents/DAggar/robosuite/data/PandaLift/intervention"]' \
  train.use_weighted_sampler=true \
  train.labeled_intervention_only=true \
  train.intervention_sample_weight=1.5 \
  train.non_intervention_sample_weight=1.0 \
  train.epochs=100 \
  save_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/flow/altogether"
