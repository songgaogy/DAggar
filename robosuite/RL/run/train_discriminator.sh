export CUDA_VISIBLE_DEVICES=7

python /home/gy/Documents/DAgger/robosuite/RL/discriminator/train.py \
  --data_root /home/gy/Documents/DAgger/robosuite/RL/data/PandaLiftImage128 \
  --train_steps step-150k,step-470k \
  --eval_steps step-960k \
  --use_images --use_state_action \
  --image_pretrained \
  --batch_size 256 --lr 1e-4 --epochs 30 \
  --val_frac 0.1 \
  --num_workers 64 \
  --logdir /home/gy/Documents/DAgger/robosuite/RL/discriminator/out