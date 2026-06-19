# Latent Dynamics Features for Failure Discrimination

`robosuite.discriminator.dyn_disc` contains a compact WAM-style pipeline:

1. train a visual dynamics model on demonstration trajectories;
2. freeze the learned latent encoder;
3. run lightweight KNN failure discriminators on top of the latent sequence.

This branch (`v0-knn`) supports two KNN discriminator variants on a frozen
DINOv3 encoder:

| Variant | Main class | Idea |
| --- | --- | --- |
| Single-bank KNN | `SingleBankBenchmarkDiscriminator` | Build a success feature bank and flag frames far from success demos. |
| Two-bank KNN | `TwoBankBenchmarkDiscriminator` | Build success and failure banks, then score each frame by relative distance to both. |

Both variants reuse the same frozen `DynEncoder` and the benchmark trajectory API.

---

## Repository layout

```text
dyn_disc/
  core/
    model_loader.py                # Rebuild/load VisualDynamicsModel checkpoints
  detectors/
    single_bank_knn.py             # DynEncoder + single-bank SingleBankKNN
    two_bank_knn.py                # Two-bank KNN detector (TwoBankKNN)
  adapters/
    single_bank.py                 # Single-bank benchmark adapter
    two_bank.py                    # Two-bank benchmark adapter
  sim_benchmark.py                 # Single-bank robosuite FailureBenchmark runner
  sim_benchmark_two_bank.py        # Two-bank robosuite FailureBenchmark runner
  training/
    train.py                       # Hydra entry for latent dynamics pretraining
  visualization/
    visualize.py                   # KNN per-trajectory visualization (single + two-bank)
    vis_robosuite_latent_dinov3.py # Latent-space diagnostic visualization
  config/                          # Hydra configs (DINOv3 encoder)
  data/                            # HDF5 / preprocessed datasets
  models/                          # DINOv3 encoder, proprio MLP, ViT predictor, dynamics model
  utils/                           # Normalization, tensor helpers
  scripts/                         # Bash entrances for train/eval/vis
```

---

## Environment

In this workspace the expected Python environment is:

```bash
/home/dodo/miniconda3/envs/dagger/bin/python
```

Bash scripts generally expose this as `PYTHON_BIN`.

Core dependencies include PyTorch, Hydra/OmegaConf, torchvision, einops, numpy, scikit-learn, matplotlib, and the local `benchmark` package.

---

## Training the dynamics model

Entry point:

```bash
python -m robosuite.discriminator.dyn_disc.training.train
```

The trainer is Hydra-configured through `config/train.yaml` and `config/env/*.yaml`.
An explicit DINOv3 encoder config is **required** — a missing `encoder:` block is
an error (there is no implicit fallback encoder on this branch).

Main data backends:

| Hydra env | Dataset |
| --- | --- |
| `env=hdf5` | `HDF5DynamicsModelDataset` for robosuite-style rollout HDF5 trees. |
| `env=preprocessed` | `PreprocessedCacheDynamicsModelDataset` for `.npz` caches. |

Checkpoint directory contents expected by downstream encoders:

```text
hydra.yaml
normalizer.pth
checkpoint(s)/model_<epoch>.pth
```

### DINOv3 WAM dynamics

The robosuite simulation DINOv3 dynamics config is:

```bash
bash robosuite/discriminator/dyn_disc/scripts/train_dinov3_robosuite_dynamics.sh
```

The current setup is intentionally asymmetric:

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

The proprio/action encoders are MLPs:

```text
proprio_encoder: 14 -> 64 -> 64
action_encoder:  56 -> 128 -> 64
```

The dynamics predictor is kept near a 60M-parameter trainable budget by using a
9-layer ViT predictor.

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

The two-bank adapter defaults to `feature_source=transformer` and `transformer_layer=1`.

---

## Discriminator variants

### Single-bank KNN

Files:

- `detectors/single_bank_knn.py`
- `adapters/single_bank.py`
- `sim_benchmark.py`
- `scripts/run_single_bank_robosuite_benchmark.sh`

Workflow:

1. Split success trajectories per task into success bank and success calibration sets.
2. Build a KNN memory bank from success-bank latent frames.
3. Score each frame by minimum weighted L2 distance to the success bank.
4. Calibrate a per-task threshold on held-out success frames:

```text
tau = percentile(success_calib_scores, 100 - delta)
pred_t = 1 iff score_t >= tau
```

Simulator benchmark entrance:

```bash
MODEL_CKPT=/abs/path/to/checkpoint/model_50.pth \
bash robosuite/discriminator/dyn_disc/scripts/run_single_bank_robosuite_benchmark.sh
```

Important env knobs:

```bash
TASKS="PickPlaceCereal"
MAX_FAIL_PER_TASK=100
MAX_SUCCESS_PER_TASK=100
KNN_FEATURE_SOURCE=encoder      # encoder or transformer
KNN_TRANSFORMER_LAYER=-1
DELTA=10.0
CALIB_FRACTION=0.2
```

### Two-bank KNN

Files:

- `detectors/two_bank_knn.py`
- `adapters/two_bank.py`
- `sim_benchmark_two_bank.py`
- `scripts/run_two_bank_robosuite_benchmark.sh`

Workflow:

1. Build the success bank + success-calibration split from the success eval set.
2. Discover a disjoint GT failure bank from a labeled failure split
   (`fail_rollout-labeled`) that is `video_id`-disjoint from the eval failure set.
3. Slice each failure-bank trajectory from `first_gt_failure_frame()` onward.
4. Score each frame with one of:

```text
d_succ = min distance to success bank
d_fail = min distance to failure bank

difference: score = d_succ - alpha * d_fail
ratio:      score = d_succ / (d_succ + d_fail + eps)
dsucc_only: score = d_succ
```

5. Calibrate with `success_percentile` or `two_class_youden`.

Simulator benchmark entrance:

```bash
MODEL_CKPT=/abs/path/to/checkpoint/model_50.pth \
bash robosuite/discriminator/dyn_disc/scripts/run_two_bank_robosuite_benchmark.sh
```

Important env knobs:

```bash
TASKS="PickPlaceCereal"
FAIL_BANK_PER_TASK=25
FAIL_BANK_LAST_K=60
SCORE_MODE=difference           # difference, ratio, dsucc_only
ALPHA=1.0
CALIB_MODE=success_percentile   # success_percentile or two_class_youden
KNN_FEATURE_SOURCE=transformer
KNN_TRANSFORMER_LAYER=1
```

The runner and the adapter both hard-assert disjointness between eval
trajectories, failure-bank trajectories, and failure-calibration trajectories by
`video_id`.

---

## Per-trajectory visualization

Each KNN variant renders, for a sampled set of failure trajectories, an MP4
(per-frame HUD + red border on predicted-failure frames) plus a multi-page PDF
(per-trajectory score curve, calibrated threshold, GT failure segments, summary
page).

### Single-bank KNN visualization

```bash
MODEL_CKPT=/abs/path/to/checkpoint/model_50.pth \
TASK=PickPlaceCereal \
  bash robosuite/discriminator/dyn_disc/scripts/visualize_single_bank_robosuite.sh
```

### Two-bank KNN visualization

```bash
MODEL_CKPT=/abs/path/to/checkpoint/model_50.pth \
TASK=PickPlaceCereal \
  bash robosuite/discriminator/dyn_disc/scripts/visualize_two_bank_robosuite.sh
```

Both wrap `visualization/visualize.py`; the two-bank path is selected with
`--mode two_bank`.

---

## Public Python API

```python
from robosuite.discriminator.dyn_disc import (
    DetectionResult,
    DynEncoder,
    SingleBankBenchmarkDiscriminator,
    SingleBankKNN,
    TwoBankBenchmarkDiscriminator,
    TwoBankKNN,
    knn_min_l2_dist,
    load_model,
)
```

Use the benchmark adapters when working with `benchmark.core.BenchmarkTrajectory`.
Use `DynEncoder`, `SingleBankKNN`, or `TwoBankKNN` directly for custom data
loaders or online scoring.

Recommended explicit imports for new code:

```python
from robosuite.discriminator.dyn_disc.adapters import (
    SingleBankBenchmarkDiscriminator,
    TwoBankBenchmarkDiscriminator,
)
from robosuite.discriminator.dyn_disc.detectors import (
    DynEncoder,
    SingleBankKNN,
    TwoBankKNN,
)
```

---

## Implementation notes

- Keep checkpoint tensors, `hydra.yaml`, and `normalizer.pth` together. `DynEncoder` depends on that layout.
- `camera_to_view` maps real camera names to training view names, for example `agentview:agentview`.
- Larger `delta` lowers the calibration percentile and usually makes detectors stricter.
- The two-bank failure-bank method requires usable `first_gt_failure_frame()` annotations.
- The benchmark adapters pad scores/predictions back to `trajectory.num_frames` when encoded feature length is shorter than the raw trajectory length.
```
