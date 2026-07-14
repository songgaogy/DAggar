export CUDA_VISIBLE_DEVICES=0

python /home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/robosuite/policy/generate_data.py \
    ckpt="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/checkpoints/PandaLift/flow/BC_warmup/flow_policy_ep0040_20260303_135632.pt" \
    max_steps=250 \
    generate.output_dir="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/data/PandaLift/unlabled" \
    generate.episodes=50 \
    generate.failure_only=true
