# Policy Features for Failure Discrimination

This branch is a policy-encoder ablation for the nnPU discriminator. It supports
only this pipeline:

```text
existing flow_multi-update EMA policy
  -> frozen 256D task_scene_cond feature
  -> shared nnPU MLP head
  -> per-task success-percentile threshold
```

There is no dynamics-model training or DINO/dynamics feature source in this
branch. The default policy checkpoint is
`checkpoints/multitask_6/flow_multi_ep0100.pt`.

## Method

The policy encoder reproduces policy evaluation preprocessing: three checkpoint
cameras, center-square crop and resize to 128x128, ImageNet normalization,
task-specific qpos/qvel extraction and checkpoint proprio normalization, and
the first checkpoint prompt for each task. It strictly loads `ema_model`, stays
frozen in evaluation mode, and emits only the fused `task_scene_cond` feature.
All policy and discriminator tensor computation is CUDA-only and float32.

The nnPU head treats frames as either positive or unlabeled:

- Positive `P`: pre-completion frames from success rollouts.
- Unlabeled `U`: whole failure rollouts, without using failure onset.

For success prior `pi_p`, the non-negative PU risk is:

```text
R_pu = pi_p E_p[loss(+1, g)]
     + max(0, E_u[loss(-1, g)] - pi_p E_p[loss(-1, g)])
```

The head produces success-likeness logit `g(z)` and reports
`failure_score = -g(z)`. Each task threshold is the `(100 - delta)` percentile
of held-out success failure scores. Data discovery, trajectory-disjoint splits,
sampling, nnPU loss, calibration, metrics, and seed semantics are unchanged
from the previous benchmark.

## Run the benchmark

```bash
TASKS="PickPlaceCereal PickPlaceMilk" PI_P=0.5 \
  bash robosuite/discriminator/dyn_disc/scripts/run_pu_bce_robosuite_benchmark.sh
```

The runner defaults to the policy checkpoint above and writes an isolated run:

```text
checkpoints/dyn_disc/ablations/policy/runs/run_<timestamp>_<tasks>/
  benchmark.json
  unlabeled_pool_manifest.json
  checkpoints/pu_bce_head.pth
  visualizations/
```

The manifest and head checkpoint record the policy SHA256, `ema_model`, feature
name, float32 dtype, camera order, prompt, and preprocessing version. A loaded
head is rejected when this feature contract does not match.

Important controls:

```text
POLICY_CKPT             frozen flow_multi-update checkpoint
TASKS                   space-separated robosuite tasks
PI_P                    positive class prior
LOSS_SURROGATE          logistic or sigmoid
DELTA                   success false-alarm budget in percent
ENCODE_BATCH_SIZE       policy inference batch size
PRELOAD_WORKERS         trajectory preload workers
PREFETCH_FACTOR         batches prefetched per worker
PIN_MEMORY              1 enables pinned host buffers
FEATURE_CACHE_DIR       persistent frame-feature cache
REUSE_FEATURE_CACHE     1 reuses valid cached features
```

The cache key includes the policy checkpoint hash, EMA selection, task and
prompt, camera order, preprocessing version, source trajectory identity, and
success-prefix length. This permits repeat experiments without silently reusing
features from an incompatible policy or preprocessing contract.

## Visualization

Render trajectory videos and score PDFs with the existing visualization
launcher. A fitted head checkpoint can be supplied to skip training.

Visualize the policy latent space with:

```bash
TASK=Stack \
  bash robosuite/discriminator/dyn_disc/scripts/vis_policy_latent_robosuite.sh
```

This produces PCA and t-SNE PNGs, a compressed NPZ with original features and
coordinates, and JSON metadata under the run's `visualizations/latent/`
directory. Success, failure-before-GT, and failure-after-GT colors are for
offline inspection only; GT onset never enters training or calibration.

## Python API

The main interfaces are:

```python
from robosuite.discriminator.dyn_disc.adapters import (
    PolicyBenchmarkDiscriminator,
    PUBCEBenchmarkDiscriminator,
)
from robosuite.discriminator.dyn_disc.detectors import (
    PolicyFeatureEncoder,
    PUBCEDiscriminator,
    pu_risk,
)
```

Use `PolicyFeatureEncoder` for batched multimodal policy features,
`PolicyBenchmarkDiscriminator` for trajectory loading and persistent caching,
and `PUBCEBenchmarkDiscriminator` for fitting, calibration, and scoring.

## Tests

Run targeted tests with the repository environment:

```bash
/home/dodo/miniconda3/envs/dagger/bin/python -m pytest \
  robosuite/discriminator/dyn_disc/tests/test_policy_encoder.py \
  robosuite/discriminator/dyn_disc/tests/test_pu_bce_discriminator.py -v
```

Tensor tests require CUDA. They skip rather than execute tensor computation on
CPU when CUDA is unavailable.
