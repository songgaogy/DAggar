export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="offline"
export WANDB_ENTITY="songgao-personal"

python train.py \
  --config-name=train_diffusion_unet_hybrid_workspace \
  task=pandalift_image \
  task.dataset_path="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert" \
  task.dataset.dataset_path="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert" \
  training.device=cuda:0 \
  training.seed=42
