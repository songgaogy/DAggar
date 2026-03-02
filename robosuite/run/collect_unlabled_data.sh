export CUDA_VISIBLE_DEVICES=0

python /home/dodo/Documents/DAggar/robosuite/robosuite/policy/generate_data.py \
    ckpt="/home/dodo/Documents/DAggar/robosuite/checkpoints/flow/BC_warmup/flow_policy_ep0010_20260301_222307.pt" \
    max_steps=250 \
    generate.output_dir="/home/dodo/Documents/DAggar/robosuite/data/PandaLift/unlabled" \
    generate.episodes=50