export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE="offline"
export WANDB_ENTITY="songgao-personal"

python -m dyn_model.train \
    env=pandalift \
    env.train_data_path='[/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert,/home/dodo/Documents/DAggar/robosuite/data/PandaLift/unlabled]' \
    env.val_data_path=/home/dodo/Documents/DAggar/robosuite/data/PandaLift/expert \
    env.policy_ckpt_path="/home/dodo/Documents/DAggar/robosuite/lpb_robosuite/data/outputs/2026.03.06/13.42.42_train_diffusion_unet_hybrid_pandalift_image/checkpoints/50.ckpt"
