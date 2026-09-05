# DSRL-NA pipeline

This directory contains the CUDA-only DSRL-NA offline-to-online pipeline for
the frozen `flow_multi-update` base policy. The current task configuration is
`PickPlaceCereal`.

## Data and checkpoints

- Base policy: `checkpoints/multitask_6/flow_multi_ep0100.pt`
- Offline replay: `data/PickPlaceCereal/offline_data-vast/vast_offline_transitions.pt`
- DINOv2: `~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`
- Shared cache: `outputs/baseline/dsrl/cache/PickPlaceCereal/`
- Runs: `outputs/baseline/dsrl/PickPlaceCereal/PickPlaceCereal_<timestamp>/`

The source replay is a large legacy pickle. Its first conversion loads the
file once and writes compact memory-mapped frozen DINOv2 and flow features.
Subsequent runs validate the fingerprint and reuse that cache.

## Commands

Build or validate the cache:

```bash
bash robosuite/pipeline/scripts/build_cache.sh
```

Run the approved four-worker, five-episode-per-worker training job:

```bash
bash robosuite/pipeline/scripts/train.sh
```

Evaluate a DSRL checkpoint deterministically for 50 episodes, with a maximum
of 500 primitive steps per episode:

```bash
CKPT=/path/to/checkpoints/latest.pt \
  bash robosuite/pipeline/scripts/eval_policy.sh
```

`NUM_EPISODES`, `MAX_STEPS`, and `DEVICE` are optional environment overrides.
The task and checkpoint episode are read from the checkpoint. Results are
written to the owning run's `eval/episode_<completed_episodes>/` directory.

Hydra overrides can be appended to the cache and training commands. Evaluation
accepts standard CLI arguments after the launcher. Tensor computation is never
allowed to fall back to CPU.
