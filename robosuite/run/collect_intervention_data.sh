export CUDA_VISIBLE_DEVICES=1

python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/collect_human_intervention.py \
    intervention.cool_down_second=1.5 \
    intervention.directory="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/intervention" \
    ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/PandaLift/flow/BC_warmup/flow_policy_ep0040_20260303_135632.pt"
