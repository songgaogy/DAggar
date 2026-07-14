export CUDA_VISIBLE_DEVICES=0

python /home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/robosuite/discriminator/finetune_discriminator.py \
  policy_ckpt="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/checkpoints/PandaLift/flow/BC_warmup/flow_policy_ep0040_20260303_135632.pt" \
  disc_ckpt="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/checkpoints/PandaLift/discriminator/bce/bce_discriminator_best.pt" \
  data_dir="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/data/PandaLift/intervention" \
  save_dir="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/checkpoints/PandaLift/discriminator/bce/finetune" \
  train_eps=-1 \
  labels.window_before=5 \
  labels.window_after=2 \
  train.epochs=20 \
  eval.threshold=0.7 \
  eval.eval_ratio=0.2
