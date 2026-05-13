# LPB v2 - Latent Dynamics Features for Failure Discrimination

`robosuite.discriminator.lpb_v2` contains a compact LPB/WAM-style pipeline:

1. train a visual dynamics model on demonstration trajectories;
2. freeze the learned latent encoder;
3. run lightweight failure discriminators on top of the latent sequence.

The current folder supports three discriminator variants:

| Variant | Main class | Idea |
| --- | --- | --- |
| Single-bank KNN | `LPBV2BenchmarkDiscriminator` | Build a success feature bank and flag frames far from success demos. |
| Two-bank KNN | `TwoBankBenchmarkDiscriminator` | Build success and failure banks, then score each frame by relative distance to both. |
| BCE head | `BCEBenchmarkDiscriminator` | Train one shared MLP head on expert-like vs failure-suffix latent frames, then calibrate per-task thresholds. |

All variants reuse the same frozen LPB v2 encoder and benchmark trajectory API.

---

## Repository layout

```text
lpb_v2/
  core/
    model_loader.py                # Rebuild/load VisualDynamicsModel checkpoints
  detectors/
    single_bank_knn.py             # LPBV2Encoder + single-bank LPBV2KNN
    two_bank_knn.py                # Two-bank KNN detector
    bce.py                         # BCEHead + BCEDiscriminator
  adapters/
    single_bank.py                 # Single-bank benchmark adapter
    two_bank.py                    # Two-bank benchmark adapter
    bce.py                         # BCE benchmark adapter
  sim_benchmark.py                 # Simulator FailureBenchmark runner
  real_world_single_bank.py        # Real-world Agilex single-bank runner
  real_world_two_bank.py
  real_world_bce.py
  training/
    train.py                       # Hydra entry for latent dynamics pretraining
  visualization/
    visualize.py                   # Latent visualization utilities
  config/                          # Hydra configs
  data/                            # HDF5 / preprocessed / Agilex datasets
  models/                          # ResNet, proprio MLP, ViT predictor, dynamics model
  utils/                           # Normalization, tensor helpers, latent plotting
  explore/
    diagnose_latent_separability.py
    run_diagnose_latent_separability.bash
  scripts/                         # Bash entrances for train/eval/vis
  tests/
    test_bce_discriminator.py
```

---

## Environment

In this workspace the expected Python environment is:

```bash
/home/dodo/miniconda3/envs/daggar/bin/python
```

Bash scripts generally expose this as `PYTHON_BIN`. For runs that log through WandB, use:

```bash
export WANDB_MODE=offline
export WANDB_NAME=songgao-personal
```

Core dependencies include PyTorch, Hydra/OmegaConf, torchvision, einops, numpy, scikit-learn, matplotlib, and the local `benchmark` package.

---

## Training the LPB v2 dynamics model

Entry point:

```bash
python -m robosuite.discriminator.lpb_v2.training.train
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
bash robosuite/discriminator/lpb_v2/scripts/train_lpb_v2_agilex_dynamics.sh
```

Useful overrides:

```bash
TASKS="candy_in_plate duck_in_bowl" \
EPOCHS=50 \
BATCH_SIZE=256 \
FRAMESKIP=1 \
bash robosuite/discriminator/lpb_v2/scripts/train_lpb_v2_agilex_dynamics.sh
```

The script assumes a built Agilex cache. It points at `data/.agilex_train_cache` by default and writes under `checkpoints/lpb_v2/dynamics/<run-name>-<timestamp>/`.

Simulator/preprocessed-cache training entrance:

```bash
TASK=PickPlaceCan \
bash robosuite/discriminator/lpb_v2/scripts/train_lpb_v2_dynamics.sh
```

This script uses `env=preprocessed`, defaults to caches under `data/.lpb_score_preprocessed_cache`, and accepts Hydra overrides through trailing CLI arguments.

---

## Feature extraction

`LPBV2Encoder` loads:

- the dynamics checkpoint;
- sibling `hydra.yaml`;
- sibling `normalizer.pth`.

It applies the same image/state/action normalization and view mapping used in training, then emits frame-level features.

Supported feature spaces:

| `feature_source` | Feature |
| --- | --- |
| `encoder` | Concatenated `[visual_emb; proprio_emb; action_emb]`. Block weights apply to visual/proprio/action dimensions. |
| `transformer` | Flattened ViT hidden state at `transformer_layer`. Distances use uniform L2 in benchmark adapters. |

For current real-world scripts, two-bank and BCE default to `feature_source=transformer` and `transformer_layer=1`.

---

## Discriminator variants

### Single-bank KNN

Files:

- `detectors/single_bank_knn.py`
- `adapters/single_bank.py`
- `real_world_single_bank.py`
- `sim_benchmark.py`
- `scripts/run_lpb_v2_real_world_benchmark.sh`

Workflow:

1. Split success trajectories per task into success bank and success calibration sets.
2. Build a KNN memory bank from success-bank latent frames.
3. Score each frame by minimum weighted L2 distance to the success bank.
4. Calibrate a per-task threshold on held-out success frames:

```text
tau = percentile(success_calib_scores, 100 - delta)
pred_t = 1 iff score_t >= tau
```

Run:

```bash
MODEL_CKPT=/abs/path/to/checkpoints/model_49.pth \
bash robosuite/discriminator/lpb_v2/scripts/run_lpb_v2_real_world_benchmark.sh
```

Simulator benchmark entrance:

```bash
MODEL_CKPT=/abs/path/to/checkpoints/model_49.pth \
bash robosuite/discriminator/lpb_v2/scripts/run_lpb_v2_benchmark.sh
```

Important env knobs:

```bash
TASKS="candy_in_plate duck_in_bowl"
MAX_FAIL_PER_TASK=25
MAX_SUCCESS_PER_TASK=25
KNN_FEATURE_SOURCE=encoder      # encoder or transformer
KNN_TRANSFORMER_LAYER=-1
DELTA=10.0
CALIB_FRACTION=0.2
```

Note: the current single-bank scripts assign `MODEL_CKPT` inside the script body. If that assignment is still present, edit it or pass `--model-ckpt` directly to the Python runner.

### Two-bank KNN

Files:

- `detectors/two_bank_knn.py`
- `adapters/two_bank.py`
- `real_world_two_bank.py`
- `scripts/run_two_bank_real_world_benchmark.sh`

Workflow:

1. Build the success bank from success trajectories.
2. Select a disjoint failure bank from labeled failure trajectories not present in the eval set.
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

Run:

```bash
MODEL_CKPT=/abs/path/to/checkpoints/model_49.pth \
bash robosuite/discriminator/lpb_v2/scripts/run_two_bank_real_world_benchmark.sh
```

Important env knobs:

```bash
FAIL_BANK_PER_TASK=10
SCORE_MODE=difference           # difference, ratio, dsucc_only
ALPHA=1.0
CALIB_MODE=success_percentile   # success_percentile or two_class_youden
KNN_FEATURE_SOURCE=transformer
KNN_TRANSFORMER_LAYER=1
```

The adapter asserts disjointness between eval trajectories, failure-bank trajectories, and failure-calibration trajectories by `video_id`.

### BCE discriminator

Files:

- `detectors/bce.py`
- `adapters/bce.py`
- `real_world_bce.py`
- `scripts/run_bce_real_world_benchmark.sh`
- `tests/test_bce_discriminator.py`

Workflow:

1. Encode success trajectories and split them into train/calibration per task.
2. For each disjoint failure-bank trajectory, use ground-truth failure timing:

```text
prefix [0, first_gt_failure_frame)  -> expert-like set D_e
suffix [first_gt_failure_frame, T)  -> other/failure set D_o
```

3. Train one shared `BCEHead` with `BCEWithLogitsLoss`.
4. Convert expert-likeness logits into failure scores:

```text
g(z) = head(z)                 # larger means more expert-like
failure_score = -g(z)          # larger means more failure-like
```

5. Calibrate per-task thresholds on held-out success frames:

```text
tau_task = percentile(success_calib_failure_scores, 100 - delta)
pred_t = 1 iff failure_score_t >= tau_task
```

Run:

```bash
MODEL_CKPT=/abs/path/to/checkpoints/model_49.pth \
bash robosuite/discriminator/lpb_v2/scripts/run_bce_real_world_benchmark.sh
```

Important env knobs:

```bash
FAIL_BANK_PER_TASK=25
HEAD_HIDDEN=256
HEAD_LAYERS=2
EPOCHS=20
LR=3e-4
WEIGHT_DECAY=1e-4
BATCH_SIZE=512
MAX_EXPERT_OTHER_RATIO=1.0     # <=0 disables the D_e cap
SAVE_CKPT_DIR=/path/to/out/checkpoints
KNN_FEATURE_SOURCE=transformer
KNN_TRANSFORMER_LAYER=1
```

Hard invariant: `BCEBenchmarkDiscriminator.fit_on_benchmark(...)` trains and calibrates only. It does not call `bench.evaluate(...)` and does not compute AUROC during training. Evaluation happens later in `real_world_bce.py`.

---

## Latent separability diagnostic

Diagnostic entry:

```bash
bash robosuite/discriminator/lpb_v2/explore/run_diagnose_latent_separability.bash
```

It can:

- encode all benchmark frames into `latents.npz`;
- save metadata into `latents_meta.json`;
- run pooled MMD, Ledoit-Wolf Mahalanobis AUROC, and matched-timestep AUROC;
- emit `separability_summary.json`, ROC plots, and histograms.

To cache features only:

```bash
CACHE_FEATURES_ONLY=1 \
bash robosuite/discriminator/lpb_v2/explore/run_diagnose_latent_separability.bash
```

To re-analyze a previous cache without GPU encoding:

```bash
LOAD_CACHE=/path/to/latents.npz \
bash robosuite/discriminator/lpb_v2/explore/run_diagnose_latent_separability.bash
```

---

## Public Python API

```python
from robosuite.discriminator.lpb_v2 import (
    BCEDiscriminator,
    BCEBenchmarkDiscriminator,
    BCEHead,
    DetectionResult,
    LPBV2BenchmarkDiscriminator,
    LPBV2Encoder,
    LPBV2KNN,
    TwoBankBenchmarkDiscriminator,
    TwoBankKNN,
    load_model,
)
```

Use the benchmark adapters when working with `benchmark.core.BenchmarkTrajectory`. Use `LPBV2Encoder`, `LPBV2KNN`, `TwoBankKNN`, or `BCEDiscriminator` directly for custom data loaders or online scoring.

Recommended explicit imports for new code:

```python
from robosuite.discriminator.lpb_v2.adapters import (
    BCEBenchmarkDiscriminator,
    LPBV2BenchmarkDiscriminator,
    TwoBankBenchmarkDiscriminator,
)
from robosuite.discriminator.lpb_v2.detectors import (
    BCEDiscriminator,
    LPBV2Encoder,
    LPBV2KNN,
    TwoBankKNN,
)
```

---

## Tests

BCE unit tests:

```bash
/home/dodo/miniconda3/envs/daggar/bin/python -m pytest \
  robosuite/discriminator/lpb_v2/tests/test_bce_discriminator.py -v
```

The tests cover head shape, synthetic separability, deterministic thresholding, scoring output, disjointness checks, class-balance capping, and state-dict roundtrip.

---

## Implementation notes

- Keep checkpoint tensors, `hydra.yaml`, and `normalizer.pth` together. `LPBV2Encoder` depends on that layout.
- Real-world scripts default to right-arm Agilex slices: `qpos[:, 7:14]` and `action[:, 7:14]`.
- `camera_to_view` maps real camera names to training view names, for example `cam_high:agentview`.
- Larger `delta` lowers the calibration percentile and usually makes detectors stricter.
- Failure-bank based methods require usable `first_gt_failure_frame()` annotations.
- The benchmark adapters pad scores/predictions back to `trajectory.num_frames` when encoded feature length is shorter than the raw trajectory length.
