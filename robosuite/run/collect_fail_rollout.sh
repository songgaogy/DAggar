export CUDA_VISIBLE_DEVICES=1

python /home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/robosuite/policy/collect_fail_rollout.py \
    fail_rollout.directory="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/data/PandaLift/fail_rollout" \
    fail_rollout.target_failures=50 \
    fail_rollout.max_steps=300 \
    ckpt="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/checkpoints/PandaLift/flow/BC_warmup/flow_policy_ep0040_20260303_135632.pt"
