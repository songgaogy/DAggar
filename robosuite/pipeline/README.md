# WidowX-style DSRL-SAC pipeline

This directory contains the CUDA-only DSRL-SAC pipeline for the frozen
`flow_multi-update` base policy. `PickPlaceCereal` uses the WidowX
Pick-and-Place hyperparameters from the DSRL paper while retaining the base
policy's native 8-step action chunks.

## Training setup

- Base policy: `checkpoints/multitask_6/flow_multi_ep0100.pt`
- State: frozen base-policy ResNet tokens from all policy cameras plus normalized proprioception
- Reward: sparse binary success (`0` otherwise, `1` on success)
- Warmup: 20 base-policy episodes with clipped standard-Gaussian latent noise
- Online budget: 180 episodes, for 200 total including warmup
- Parallelism: 4 spawned robosuite environments
- Learner: 30 SAC updates every 2 vector macro-steps
- Checkpoints: global episodes 20, 40, ..., 200
- Logging: JSONL, console log, and TensorBoard

The shared warmup cache is stored under
`outputs/baseline/dsrl/warmup_cache/<task>/<fingerprint>/`. Its fingerprint
includes the task, fixed warmup seed, base checkpoint, environment metadata,
cameras, flow sampling settings, feature schema, and reward schema. Training
seeds reuse this immutable task cache. `offline_data-vast` and DINOv2 are not
used by DSRL-SAC.

## Commands

Build or validate the warmup cache:

```bash
bash robosuite/pipeline/scripts/build_cache.sh
```

This command performs environment data generation on a cache miss. Obtain the
required experiment approval before running it.

Train after the cache exists (training also builds a missing cache):

```bash
bash robosuite/pipeline/scripts/train.sh
```

Evaluate a checkpoint deterministically:

```bash
/home/dodo/miniconda3/envs/dagger/bin/python -m robosuite.pipeline.eval_policy \
  --checkpoint /path/to/checkpoints/latest.pt
```

`--num-episodes`, `--max-steps`, and `--device` are optional evaluation
overrides. The existing user-maintained `scripts/eval_policy.sh` is not
modified by this migration. Hydra overrides may be appended to cache and
training commands. Tensor computation never falls back to CPU.

Old DSRL-NA checkpoints and VAST/DINO feature caches are intentionally left in
place, but they are not compatible with the DSRL-SAC checkpoint schema.
