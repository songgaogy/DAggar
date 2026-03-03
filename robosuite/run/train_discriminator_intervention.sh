export CUDA_VISIBLE_DEVICES=0

python /home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/train_discriminator_intervention.py \
  policy_ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/flow/BC_warmup/flow_policy_ep0040_20260303_135632.pt" \
  data.intervention_dir="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/intervention" \
  data.expert_dir="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert" \
  save_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/discriminator/bce/intervention_train" \
  train_eps=-1 \
  labels.window_before=5 \
  labels.window_after=2 \
  train.epochs=50 \
  eval.threshold=0.7
