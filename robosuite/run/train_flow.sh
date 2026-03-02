export CUDA_VISIBLE_DEVICES=1

python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/train_flow.py \
  data_dir="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert" \
  train.epochs=200 \
  save_dir="/home/dodo/Documents/DAggar/robosuite/checkpoints/flow/BC_warmup" \
  uf_low=0.05 \
  uf_high=0.3