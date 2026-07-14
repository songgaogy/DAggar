export CUDA_VISIBLE_DEVICES=0

python -m robosuite.discriminator.eval_discriminator \
  data.expert_dir="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/data/PandaLift/expert" \
  data.fail_rollout_dir="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/data/PandaLift/fail_rollout" \
  data.success_rollout_dir="" \
  data.obs_key="states" \
  data.camera_name="agentview" \
  policy.ckpt="/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/checkpoints/PandaLift/flow/BC_warmup/flow_policy_ep0040_20260303_135632.pt" \
  policy.device="cuda" \
  policy.image_size=128 \
  split.val_ratio=0.2 \
  labels.fail_tail_ratio=0.2 \
  float.sinkhorn_reg=0.05 \
  float.max_iter=300 \
  float.ta=8 \
  float.to=2 \
  float.num_expert_candidates=50 \
  calibration.delta=10.0 \
  online.adaptive_delta=false
