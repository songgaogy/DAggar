export CUDA_VISIBLE_DEVICES=0

python -m robosuite.discriminator.eval_discriminator \
  disc_ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/discriminator/bce/bce_discriminator_best.pt" \
  policy_ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/flow/BC_warmup/flow_policy_ep0040_20260303_135632.pt" \
  data_dir="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/intervention" \
  eval.window_before=5 \
  eval.window_after=2 \
  eval.threshold=0.7 \
  eval.batch_size=1024
