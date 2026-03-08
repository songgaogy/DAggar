export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="offline"
export WANDB_ENTITY="songgao-personal"

# python train.py \
#   --config-name=train_diffusion_unet_hybrid_workspace \
#   task=pandalift_image \
#   task.dataset_path="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert" \
#   task.dataset.dataset_path="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert" \
#   training.device=cuda:0 \
#   training.seed=42


python train.py \
  --config-dir=. \
  --config-name=image_transport_diffusion_policy_cnn.yaml \
  training.seed=42 \
  training.device=cuda:0 \
  hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' \
  dataset.dataset_path="/home/dodo/Documents/DAggar/robosuite/lpb_robosuite/data/transport/data/expert_demonstration"