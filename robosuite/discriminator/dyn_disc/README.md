# Latent Dynamics Features for Failure Discrimination

`robosuite.discriminator.dyn_disc` contains a compact dynamics-latent pipeline:

1. train a visual dynamics model (DINOv3 encoder) on demonstration trajectories;
2. freeze the learned latent encoder;
3. run a lightweight failure discriminator on top of the latent sequence.

This branch (`v0-pu-bce`) ships a single discriminator variant:

| Variant | Main class | Idea |
| --- | --- | --- |
| PU-BCE (nnPU) | `PUBCEBenchmarkDiscriminator` | Train one shared MLP head with the non-negative PU risk on (positives = success frames, unlabeled = whole failure rollouts), then calibrate per-task thresholds via success_percentile. **No GT failure timing.** |

The head reuses the frozen DINOv3 dynamics encoder (`DynEncoder`) and the shared
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
    model_loader.py                # Rebuild/load VisualDynamicsModel checkpoints
  detectors/
    single_bank_knn.py             # DynEncoder (shared frozen encoder) + helpers
    pu_bce.py                      # BCEHead + nnPU risk + PUBCEDiscriminator
  adapters/
    single_bank.py                 # DynBenchmarkDiscriminator (encoder/cache backbone)
    pu_bce.py                      # PUBCEBenchmarkDiscriminator
  robosuite_pu_bce.py              # Robosuite benchmark runner (entry)
  training/
    train.py                       # Hydra entry for latent dynamics pretraining
  visualization/
    visualize_pu_bce.py            # PU-BCE per-trajectory visualization (robosuite)
  config/                          # Hydra configs (DINOv3 only)
  data/                            # HDF5 / preprocessed / Agilex datasets
  models/                          # DINOv3 encoder, proprio MLP, ViT predictor, dynamics model
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

Bash scripts generally expose this as `PYTHON_BIN`. For runs that log through tensorboard, use:

Core dependencies include PyTorch, Hydra/OmegaConf, torchvision, einops, numpy, scikit-learn, matplotlib, and the local `benchmark` package.

---

## Training the LPB v2 dynamics model

Entry point:

```bash
python -m robosuite.discriminator.dyn_disc.training.train
```

The trainer is Hydra-configured through `config/train.yaml` and `config/env/*.yaml`.

Main data backends:

| Hydra env | Dataset |
| --- | --- |
| `env=hdf5` | `HDF5DynamicsModelDataset` for robosuite-style rollout HDF5 trees. |
| `env=preprocessed` | `PreprocessedCacheDynamicsModelDataset` for `.npz` caches. |
| `env=agilex` | `AgilexCacheDynamicsModelDataset` for real-world Agilex cache data. |

Checkpoint directory contents expected by downstream encoders:

```text
hydra.yaml
normalizer.pth
checkpoints/model_<epoch>.pth
```

Real-world Agilex training entrance:

```bash
bash robosuite/discriminator/dyn_disc/scripts/train_dyn_disc_agilex_dynamics.sh
```

Useful overrides:

```bash
TASKS="candy_in_plate duck_in_bowl" \
EPOCHS=50 \
BATCH_SIZE=256 \
FRAMESKIP=1 \
bash robosuite/discriminator/dyn_disc/scripts/train_dyn_disc_agilex_dynamics.sh
```

The script assumes a built Agilex cache. It points at `data/.agilex_train_cache` by default and writes under `checkpoints/dyn_disc/dynamics/<run-name>-<timestamp>/`.

Simulator/preprocessed-cache training entrance:

```bash
TASK=PickPlaceCan \
bash robosuite/discriminator/dyn_disc/scripts/train_dyn_disc_dynamics.sh
```

This script uses `env=preprocessed`, defaults to caches under `data/.lpb_score_preprocessed_cache`, and accepts Hydra overrides through trailing CLI arguments.

### Visual encoder choice

For discriminator pretraining, prefer a strong frozen self-supervised visual encoder before tuning a supervised ResNet from scratch. A practical priority order is:

1. **Frozen DINOv3 / DINO-style ViT** as the first baseline.
2. **DINOv3 partial tuning** with LoRA, adapters, or the last 1-2 transformer blocks if the frozen baseline underfits.
3. **ResNet50 finetune** as a speed / memory ablation, with strict validation monitoring.

The reason is that failure detection is closer to OOD / dynamics-consistency scoring than closed-set image classification. With limited labeled failure data, a finetuned ResNet50 can overfit task, camera, background, or object shortcuts. A frozen DINO-style encoder usually gives more stable features and better transfer across tasks.

If using DINOv3, do not only use the global `CLS` token unless this is an intentional lightweight baseline. Robot failures are often local: missed grasp, object slip, collision, or bad hand-object geometry. These signals can be diluted in one global image vector. Prefer one of:

```text
Lowest cost:
  patch_tokens -> mean pool -> one visual token

Recommended:
  patch_tokens on a dense grid -> spatial pool to 4x4 or 2x2 -> P visual tokens

Highest capacity:
  full patch_tokens -> P visual tokens
```

The current `VisualDynamicsModel` already uses the shape:

```text
z: (B, T, P, D)
```

With the current ResNet encoder, `P=1` in practice. A DINOv3 encoder should expose `P>1` patch or pooled dense tokens, for example `P=16` from a 4x4 pooled token grid. Then the WAM loss predicts future local visual tokens instead of only a whole-image summary:

```text
image_t, proprio_t, action_t -> patch_tokens_{t+1}
loss = MSE(pred_patch_tokens_{t+1}, target_patch_tokens_{t+1})
```

This preserves spatial evidence that is important for frame-level failure scores. If compute is limited, start with grid-pooled dense tokens rather than full patch tokens.

### DINOv3 WAM dynamics v1

The robosuite simulation DINOv3 dynamics config is:

```bash
bash robosuite/discriminator/dyn_disc/scripts/train_dinov3_robosuite_dynamics.sh
```

The current v1 setup is intentionally asymmetric:

```text
source:
  agentview_t, robot0_eye_in_hand_t, proprio_t, action[t:t+7]

target:
  agentview_{t+8}, proprio_{t+8}
```

Main choices:

- visual encoder: frozen DINOv3 ViT-B/16 from `data/pretrained/dinov3-vitb16-pretrain-lvd1689m_80M`;
- visual tokens: DINO patch tokens spatially pooled to a 4x4 grid, so `P=16`;
- visual projection: DINO hidden tokens are projected to `382` dimensions;
- source views: `agentview` and `robot0_eye_in_hand`;
- target view: `agentview` only;
- proprio: Panda `qpos[7] + qvel[7]`, selected per robosuite task through `proprio_map`;
- action chunk: causal horizon 8, flattened as `7 * 8 = 56`;
- supervision: no action target loss (`action_loss_weight=0.0`); action is used only as a conditioning input.

The v1 proprio/action encoders are MLPs rather than one-layer projections:

```text
proprio_encoder: 14 -> 64 -> 64
action_encoder:  56 -> 128 -> 64
```

This avoids the old bottleneck where an 8-step action chunk was compressed from
56 dimensions to 7 dimensions by a single linear layer. The dynamics predictor is
kept near a 60M-parameter trainable budget by using a 9-layer ViT predictor.

---

## Feature extraction

`DynEncoder` loads:

- the dynamics checkpoint;
- sibling `hydra.yaml`;
- sibling `normalizer.pth`.

It applies the same image/state/action normalization and view mapping used in training, then emits frame-level features.

Supported feature spaces:

| `feature_source` | Feature |
| --- | --- |
| `encoder` | Concatenated `[visual_emb; proprio_emb; action_emb]`. Block weights apply to visual/proprio/action dimensions. |
| `transformer` | Flattened ViT hidden state at `transformer_layer`. Distances use uniform L2 in benchmark adapters. |

The PU-BCE head defaults to `feature_source=transformer` and `transformer_layer=1`.

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
KNN_FEATURE_SOURCE=transformer
KNN_TRANSFORMER_LAYER=1
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
