# TACO Features for Failure Discrimination

`robosuite.discriminator.dyn_disc` contains a compact TACO representation pipeline:

1. pretrain a TACO temporal action-driven contrastive model on demonstration trajectories;
2. freeze the learned latent encoder;
3. run a lightweight failure discriminator on top of the latent sequence.

This branch (`v0-pu-bce`) ships a single discriminator variant:

| Variant | Main class | Idea |
| --- | --- | --- |
| PU-BCE (nnPU) | `PUBCEBenchmarkDiscriminator` | Train one shared MLP head with the non-negative PU risk on (positives = success frames, unlabeled = whole failure rollouts), then calibrate per-task thresholds via success_percentile. **No GT failure timing.** |

The head reuses the frozen TACO encoder (`DynEncoder`) and the shared
benchmark adapter backbone (`DynBenchmarkDiscriminator`).

## PU-BCE method (nnPU) and how it differs from a GT-label BCE head

A GT-label BCE head needs every failure trajectory sliced at
`first_gt_failure_frame()`: the success-like prefix joins the "expert" class and
the suffix joins the "failure" class, then a binary cross-entropy classifier is
trained on those two clean classes. That requires per-frame failure-onset
annotations.

PU-BCE removes that requirement entirely. Frames are only ever labeled as
**positive (success)** or left **unlabeled**:

- **Positives `P`**: all frames from success trajectories.
- **Unlabeled `U`**: all frames from failure-rollout trajectories, taken as a
  WHOLE (no prefix/suffix split — `first_gt_failure_frame()` is never read).

The unlabeled set is the mixture `pi_p * P + (1 - pi_p) * N` where `pi_p` is the
(unknown) fraction of success-like frames inside failure rollouts, supplied as
the hyperparameter `--pi-p` (class prior; default 0.5 with a logged warning that
it should be set from domain knowledge).

The shared MLP head `g(z)` is trained with the non-negative PU risk estimator of
Kiryo et al. (2017):

```text
R_pu = pi_p * E_p[ ell(+1, g) ] + max( 0,  E_u[ ell(-1, g) ] - pi_p * E_p[ ell(-1, g) ] )
```

- `ell` is the **logistic surrogate** by default, `ell(y, g) = softplus(-y * g)`
  (`--loss-surrogate sigmoid` switches to the Kiryo `sigmoid(-y*g)`). Logistic is
  the default because the sigmoid surrogate saturates to **zero gradient** once
  logits run negative and then collapses to the trivial `risk == pi_p` solution
  under a low `pi_p` and/or a weakly-trained encoder; softplus keeps a
  non-vanishing gradient and trains robustly across `pi_p` (empirically verified
  on `PickPlaceCereal`: logistic reaches fail-AUROC ~0.91 at `pi_p=0.3` where
  sigmoid collapses to `risk=0.300`).
- The **non-negative correction** clamps the second (negative-risk) term at
  `-beta` (default `beta=0`). We implement the simple clamped variant; the
  canonical Kiryo nnPU additionally does a gradient-ascent step when that term
  goes negative (noted in `detectors/pu_bce.py::pu_risk`).

Scoring matches the BCE convention: `failure_score = -g(z)` (larger = more
failure). Calibration is **success_percentile only** — `tau = percentile(
success-calib failure scores, 100 - delta)` — because no two-class Youden rule
is available without failure labels.

---

## Repository layout

```text
dyn_disc/
  core/
    model_loader.py                # Rebuild/load TACO checkpoints
  detectors/
    single_bank_knn.py             # DynEncoder (shared frozen encoder) + helpers
    pu_bce.py                      # BCEHead + nnPU risk + PUBCEDiscriminator
  adapters/
    single_bank.py                 # DynBenchmarkDiscriminator (encoder/cache backbone)
    pu_bce.py                      # PUBCEBenchmarkDiscriminator
  robosuite_pu_bce.py              # Robosuite benchmark runner (entry)
  training/
    train.py                       # Hydra entry for TACO InfoNCE pretraining
  visualization/
    visualize_pu_bce.py            # PU-BCE per-trajectory visualization (robosuite)
  config/                          # Hydra configs (DINOv3 only)
  data/                            # HDF5 dataset and image transforms
  models/                          # Frozen DINOv3 backbone and TACO representation model
  utils/                           # Normalization, tensor helpers, latent plotting
  scripts/                         # Bash entrances for train/eval/vis
  tests/
    test_pu_bce_discriminator.py
```

---

## Environment

In this workspace the expected Python environment is:

```bash
/home/dodo/miniconda3/envs/dagger/bin/python
```

Bash scripts generally expose this as `PYTHON_BIN`. TACO writes TensorBoard
events under `<run>/tensorboard/`.

Core dependencies include PyTorch, Hydra/OmegaConf, torchvision, einops, numpy, scikit-learn, matplotlib, and the local `benchmark` package.

---

## TACO representation pretraining

Entry point:

```bash
bash robosuite/discriminator/dyn_disc/scripts/train_taco_robosuite.sh
```

The launcher uses two GPUs by default and reads
`config/train_taco_robosuite.yaml`. It keeps the seven success-rollout sources,
`frameskip=8`, `num_hist=1`, `num_pred=1`, and `max_trajectories=100`.
The source at time `t` contains both camera views and proprioception; the target
at `t+8` contains `agentview` and proprioception. Actions are the eight-step
window `a_t ... a_{t+7}`.

The representation objective is TACO InfoNCE only:

```text
g_i = G([z_t, action_latent])
h_j = stop_gradient(z_{t+8})
score(i, j) = g_i^T W h_j
```

The diagonal pair is positive, while gathered future state representations from
the global DDP batch are negatives. Logits and cross entropy run in FP32. The
DINOv3 backbone is frozen; its projection, the proprio/action encoders, and TACO
heads are trainable. This branch does not support future-latent MSE, a dynamics
predictor, or transformer-feature pretraining.

Throughput settings include BF16 autocast, TF32, fused Adam, pinned/prefetched
data loading, uint8 image transfer, frozen-DINO micro-batches, and global-key-only
DDP communication. Formal runs are written under
`checkpoints/dyn_disc/ablations/TACO/`.

Checkpoint directory contents expected by downstream encoders:

```text
hydra.yaml
normalizer.pth
tensorboard/
checkpoint/model_<epoch>.pth
```

---

## Feature extraction

`DynEncoder` loads:

- the TACO checkpoint;
- sibling `hydra.yaml`;
- sibling `normalizer.pth`.

It applies the same image/state/action normalization and view mapping used in training, then emits frame-level features.

The only supported feature space is `feature_source=encoder`: concatenated
`[dual-view DINO projected features; proprio latent; eight-step TACO action latent]`.
Block weights apply to visual, proprio, and action dimensions. TACO has no
transformer predictor feature path.

---

## Running the PU-BCE benchmark

The robosuite runner encodes the success pool + the unlabeled failure pool,
trains the nnPU head, calibrates per-task success_percentile thresholds, and
only then calls `bench.evaluate(...)`.

```bash
MODEL_CKPT=/abs/path/to/checkpoints/model_50.pth \
TASKS="PickPlaceCereal PickPlaceMilk" PI_P=0.5 \
  bash robosuite/discriminator/dyn_disc/scripts/run_pu_bce_robosuite_benchmark.sh
```

Or call the module directly:

```bash
python -m robosuite.discriminator.dyn_disc.robosuite_pu_bce \
  --model-ckpt /abs/path/to/checkpoints/model_50.pth \
  --data-root data --tasks PickPlaceCereal \
  --fail-split fail_rollout-val-labeled \
  --success-split success_rollout-val \
  --fail-train-split fail_rollout-labeled \
  --pi-p 0.5 --epochs 20
```

Important env / CLI knobs:

```bash
PI_P=0.5                 # --pi-p   : class prior; set from domain knowledge
LOSS_SURROGATE=sigmoid   # --loss-surrogate {sigmoid,logistic}
NO_NN_CORRECTION=0       # --no-nn-correction : use plain uPU instead of nnPU
BETA=0.0                 # --beta   : lower clamp for the negative-risk term
UNLABELED_PER_TASK=25    # --unlabeled-per-task : # of whole failure rollouts pooled as U
HEAD_HIDDEN=256
HEAD_LAYERS=2
EPOCHS=20
LR=3e-4
WEIGHT_DECAY=1e-4
BATCH_SIZE=512
DELTA=10.0               # success_percentile false-alarm budget %
CALIB_FRACTION=0.2
```

`PUBCEBenchmarkDiscriminator.fit_on_benchmark(...)` HARD-asserts that the
unlabeled failure pool is disjoint by `video_id` from the eval failure set, and
does **not** call `bench.evaluate(...)` or compute any metric during training.

---

## Per-trajectory visualization

The PU-BCE visualizer renders, for a sampled set of trajectories, an MP4
(per-frame HUD + red border on predicted-failure frames) plus a multi-page PDF
(per-trajectory score curve, calibrated threshold, GT failure segments for
reference, summary page). It is robosuite-only.

```bash
MODEL_CKPT=/abs/path/to/checkpoints/model_50.pth \
TASK=PickPlaceCereal SPLIT=fail_rollout NUM_TRAJS=3 PI_P=0.5 \
  bash robosuite/discriminator/dyn_disc/scripts/visualize_pu_bce_robosuite.sh
```

Driver: `visualization/visualize_pu_bce.py`. By default it fits a fresh head and
writes `pu_bce_head.pth` under `<OUT_DIR>/checkpoints/`. Pass
`LOAD_CKPT=/abs/path/to/pu_bce_head.pth` to skip training and restore the head +
per-task thresholds (pool discovery is skipped).

Threshold shown on screen is the detector's per-task **success_percentile** tau
(there is no Youden option without failure labels). Score semantics:

```text
g(z)           = head(z)                 # success/positive-likeness logit
failure_score  = -g(z)                   # shown in HUD; larger = more failure
pred_t = 1     iff failure_score_t >= tau_task
```

Output layout:

```text
<OUT_DIR>/
  videos/<video_id>.mp4
  pu_bce_scores.pdf
  checkpoints/pu_bce_head.pth    # only on cold fit; absent under LOAD_CKPT
```

---

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

Use `PUBCEBenchmarkDiscriminator` when working with
`benchmark.core.BenchmarkTrajectory`. Use `DynEncoder` or `PUBCEDiscriminator`
directly for custom data loaders or online scoring.

Recommended explicit imports for new code:

```python
from robosuite.discriminator.dyn_disc.adapters import (
    DynBenchmarkDiscriminator,
    PUBCEBenchmarkDiscriminator,
)
from robosuite.discriminator.dyn_disc.detectors import (
    BCEHead,
    DynEncoder,
    PUBCEDiscriminator,
    pu_risk,
)
```

---

## Tests

PU-BCE unit tests:

```bash
/home/dodo/miniconda3/envs/dagger/bin/python -m pytest \
  robosuite/discriminator/dyn_disc/tests/test_pu_bce_discriminator.py -v
```

The tests cover head shape, the nnPU non-negative correction (triggers and
clamps correctly, inactive when the negative-risk term is positive), synthetic
PU separability (positives vs unlabeled-with-hidden-negatives recovers a useful
ranking), deterministic thresholding, scoring output, the prior guard, the
disjointness invariant, and state-dict roundtrip.

---

## Implementation notes

- Keep checkpoint tensors, `hydra.yaml`, and `normalizer.pth` together. `DynEncoder` depends on that layout.
- `camera_to_view` maps real camera names to training view names, for example `cam_high:agentview`.
- Larger `delta` lowers the calibration percentile and usually makes the detector stricter.
- PU-BCE does **not** require `first_gt_failure_frame()` annotations: failure rollouts are pooled whole as the unlabeled set.
- `--pi-p` (class prior) should be set from domain knowledge; the 0.5 default emits a logged warning.
- The benchmark adapter pads scores/predictions back to `trajectory.num_frames` when encoded feature length is shorter than the raw trajectory length.
