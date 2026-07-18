# Latent Dynamics Failure Discriminator

`robosuite.discriminator.dyn_disc` contains the frozen dynamics encoder and the
production nnPU failure discriminator used by the robosuite pipeline. The
discriminator consumes success frames as positives and complete failure
rollouts as unlabeled data; it never uses ground-truth failure timing for
training or threshold calibration.

The head logit `g(z)` measures success likeness. Runtime scoring uses:

```text
failure_score = -g(z)
pred_fail = failure_score >= tau_task
```

Each task threshold is calibrated only from held-out success frames:

```text
tau_task = percentile(success_calibration_failure_scores, 100 - delta)
```

## Production default

The sole production default is the healthy five-task selection reported in
`robosuite/pipeline/docs/DYN_DISC_LOGIT_HEALTH_SWEEP_20260716.md`:
`cap_c5_l1e2` at epoch 1.

| Setting | Default |
| --- | ---: |
| Training epochs | `1` |
| Scheduler horizon | `20` epochs |
| Soft cap `(C, lambda, T)` | `(5, 0.01, 1)` |
| Threshold normalization | disabled |
| `pi_p` | `0.3` |
| nnPU surrogate / `beta` | logistic / `0` |
| Head | `512 x 3` |
| AdamW learning rate / weight decay | `3e-4` / `1e-4` |
| Batch size | `512` |
| Threshold false-alarm budget `delta` | `10` |
| Calibration fraction | `0.2` |
| Feature / transformer layer | transformer / `1` |
| Action chunk | enabled |
| Seed | `0` |
| Train success / failure trajectory cap per task | `50` / `50` |
| Eval success / failure trajectory cap per task | `50` / `50` |

The scheduler horizon is deliberately independent from the number of training
epochs. `epochs=1` with `scheduler_horizon_epochs=20` reproduces the first
optimization epoch of the 20-epoch sweep that produced the selected checkpoint.
The horizon must be greater than or equal to the requested training epochs.

The quadratic-cap parameters remain explicit CLI and Python parameters for
research overrides. Such overrides are not production defaults and can
invalidate the documented five-task health result.

## Run the benchmark once

Use the maintained launcher with a dynamics checkpoint:

```bash
MODEL_CKPT=/abs/path/to/checkpoints/model_10.pth \
TASKS="PickPlaceCereal" \
bash robosuite/discriminator/dyn_disc/scripts/run_pu_bce_robosuite_benchmark.sh
```

The launcher uses `/home/dodo/miniconda3/envs/dagger/bin/python` and CUDA by
default. It trains one nnPU head, saves one canonical `pu_bce_head.pth`, runs the
benchmark once after fitting, and writes the benchmark JSON, split/provenance
manifest, TensorBoard events, and final logit-health diagnostics under the run
directory.

The equivalent Python entry point is:

```bash
/home/dodo/miniconda3/envs/dagger/bin/python -m \
  robosuite.discriminator.dyn_disc.robosuite_pu_bce \
  --model-ckpt /abs/path/to/checkpoints/model_10.pth \
  --tasks PickPlaceCereal \
  --save-ckpt-dir /abs/path/to/output/checkpoints \
  --save-json /abs/path/to/output/benchmark.json \
  --tensorboard-dir /abs/path/to/output/tensorboard
```

Run the entry point separately for each task when reproducing the selected
five-task bundle; the sweep selected five task-specific heads under one shared
configuration, not one head jointly trained across all five tasks.

The defaults already encode the selected protocol. The most important explicit
override pair is:

```bash
--epochs 1 --scheduler-horizon-epochs 20
```

The effective-logit cap uses the quadratic hinge
`mean(relu(abs(g) - c)^2)` with `--quadratic-cap-c 2` and
`--quadratic-cap-lambda 1e-2` by default. TensorBoard records lightweight
training scalars after every optimizer step under `train_step/*`; epoch-level
calibration and logit-pool diagnostics are stored under `train_epoch/*` at the
last global step of each epoch.

Training and evaluation pools are disjoint by `video_id`. Failure rollouts are
used whole as the unlabeled pool, and benchmark evaluation starts only after
training and calibration finish.

## Dynamics pretraining

The Hydra entry point for the frozen dynamics model is:

```bash
/home/dodo/miniconda3/envs/dagger/bin/python -m \
  robosuite.discriminator.dyn_disc.training.train
```

The maintained DINOv3 robosuite launcher is:

```bash
bash robosuite/discriminator/dyn_disc/scripts/train_dinov3_robosuite_dynamics.sh
```

Downstream loading requires the model checkpoint, its sibling `hydra.yaml`, and
`normalizer.pth` to remain together.

## Repository layout

```text
dyn_disc/
  adapters/                 # Benchmark encoder/cache adapter and nnPU adapter
  config/                   # Dynamics-model Hydra configuration
  core/                     # Dynamics checkpoint loader
  data/                     # Robosuite and cached dynamics datasets
  detectors/                # DynEncoder, BCEHead, nnPU risk and detector
  models/                   # DINOv3 encoder and visual dynamics model
  scripts/                  # Maintained dynamics and single-benchmark launchers
  tests/                    # Core nnPU and benchmark regression tests
  training/                 # Dynamics pretraining entry point
  utils/                    # Normalization and data helpers
  visualization/            # Reusable rendering, sampling and checkpoint helpers
  robosuite_pu_bce.py       # Production single-run benchmark entry point
```

`visualization/visualize_pu_bce.py` is a library module used by the pipeline. It
does not provide a standalone fit or benchmark CLI.

## Public Python API

```python
from robosuite.discriminator.dyn_disc import (
    BCEHead,
    DetectionResult,
    DynBenchmarkDiscriminator,
    DynEncoder,
    PUBCEBenchmarkDiscriminator,
    PUBCEDiscriminator,
    PUCalibStats,
    load_model,
    pu_risk,
)
```

Use `PUBCEBenchmarkDiscriminator` for `BenchmarkTrajectory` inputs and
`PUBCEDiscriminator` for already encoded features. Existing checkpoints with a
stored nonzero `BCEHead.logit_center`, as well as legacy checkpoints without
that field, remain loadable; new production training does not apply threshold
normalization.

## Verification

The focused test suite lives under `robosuite/discriminator/dyn_disc/tests/`.
Any test that performs tensor computation must run in the `dagger` environment
with CUDA available; there is no CPU fallback.

> NOTE: if you want to tune the model output, check branch "dipole-rl/v3-vast_debug" commit "2380c7afb5e80b6af1fcedfba0b3620ceafa2e99". You can also refer to [this report](./DYN_DISC_LOGIT_HEALTH_SWEEP_20260716.md)
