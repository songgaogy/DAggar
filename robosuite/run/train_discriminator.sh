export CUDA_VISIBLE_DEVICES=1

python /home/dodo/Documents/DAggar/robosuite/robosuite/discriminator/train_discriminator.py \
  policy_ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/flow/BC_warmup/flow_policy_ep0040_20260303_135632.pt" \
  data.expert_dir="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert" \
  data.unlabeled_dir="//home/dodo/Documents/DAggar/robosuite/data/PandaLift/unlabled" \
  data.unlabled_size=100 \
  save_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/discriminator/bce" \
  train.epochs=50
