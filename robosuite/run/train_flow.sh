export CUDA_VISIBLE_DEVICES=1

python /home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/robosuite/policy/train_flow.py \
  data_dir="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/data/PandaLift/expert" \
  train.epochs=200 \
  save_dir="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/checkpoints/flow/BC_warmup"