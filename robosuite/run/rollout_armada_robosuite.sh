export CUDA_VISIBLE_DEVICES=0,1

# dummy failure detector
python /home/dodo/Documents/DAggar/robosuite_dipole-rl_v3/robosuite/armada/rollout.py \
    --config-name rollout_armada_robosuite \
    failure_detection._target_=robosuite.armada.failure_detector.dummy_detector.DummyFailureDetector \
    train.policy.num_inference_steps=20 \
