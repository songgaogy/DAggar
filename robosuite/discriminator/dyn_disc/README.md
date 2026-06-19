# Latent Dynamics Features for Failure Discrimination

`robosuite.discriminator.dyn_disc` contains a compact WAM-style pipeline:

1. train a visual dynamics model on demonstration trajectories;
2. freeze the learned latent encoder;
3. run lightweight failure discriminators on top of the latent sequence.

The current folder supports one discriminator variant:

| Variant | Main class | Idea |
| --- | --- | --- |
| BCE head | `BCEBenchmarkDiscriminator` | Train one shared MLP head on expert-like vs failure-suffix latent frames (GT failure split), then calibrate per-task thresholds. |

The discriminator reuses the frozen DINOv3 dynamics encoder and the benchmark trajectory API.

---

## Repository layout

```text
dyn_disc/
  core/
    model_loader.py                # Rebuild/load VisualDynamicsModel checkpoints
  detectors/
    encoder.py                     # DynEncoder + DetectionResult
    bce.py                         # BCEHead + BCEDiscriminator
  adapters/
    base.py                        # DynBenchmarkDiscriminator (shared encoding/cache base)
    bce.py                         # BCE benchmark adapter
  robosuite_bce.py                 # Robosuite FailureBenchmark BCE runner
  training/
    train.py                       # Hydra entry for latent dynamics pretraining
  visualization/
    visualize_bce.py               # BCE discriminator per-trajectory visualization
  config/                          # Hydra configs (DINOv3 encoder)
  data/                            # HDF5 / preprocessed datasets
  models/                          # DINOv3 encoder, proprio MLP, ViT predictor, dynamics model
  utils/                           # Normalization, tensor helpers
  scripts/                         # Bash entrances for train/eval/vis
  tests/
    test_bce_discriminator.py
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

Checkpoint directory contents expected by downstream encoders:

```text
hydra.yaml
normalizer.pth
checkpoint/model_<epoch>.pth
```

Robosuite DINOv3 dynamics training entrance:

```bash
bash robosuite/discriminator/dyn_disc/scripts/train_dinov3_robosuite_dynamics.sh
```

It accepts Hydra overrides through trailing CLI arguments and writes under `checkpoints/dyn_disc/dynamics/<run-name>-<timestamp>/`.

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

A DINOv3 encoder exposes `P>1` patch or pooled dense tokens, for example `P=16` from a 4x4 pooled token grid. Then the WAM loss predicts future local visual tokens instead of only a whole-image summary:

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

BCE defaults to `feature_source=transformer` and `transformer_layer=1`.

---

## BCE discriminator

Files:

- `detectors/encoder.py`  — `DynEncoder` + `DetectionResult`
- `detectors/bce.py`      — `BCEHead` + `BCEDiscriminator`
- `adapters/base.py`      — `DynBenchmarkDiscriminator` (shared encoding/cache base)
- `adapters/bce.py`       — `BCEBenchmarkDiscriminator`
- `robosuite_bce.py`      — robosuite FailureBenchmark runner
- `scripts/run_bce_robosuite_benchmark.sh`
- `tests/test_bce_discriminator.py`

Workflow:

1. Encode **train** success trajectories (from a split disjoint from the eval
   success split) and split them into train/calibration per task. Only the
   pre-done frames are used: each success trajectory is truncated at the first
   `is_success==True` frame (`prefix_frames_before_done()`), so post-done idle
   frames never enter `D_e` or calibration. The eval success split is used only
   by `bench.evaluate`, never for training (no train/val contamination).
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

5. Calibrate per-task thresholds. Default is **two-class Youden** (failure-aware):

```text
tau_task = argmax_t  TPR(t) - FPR(t)
           over s_succ = success-calib failure scores
           and  s_fail = fail-bank GT-suffix failure scores
pred_t = 1 iff failure_score_t >= tau_task
```

Selectable via `--calib-mode`. The percentile alternative is:

```text
# --calib-mode success_percentile
tau_task = percentile(success_calib_failure_scores, 100 - delta)
```

Calibration only affects `pred_t` (and downstream F1 / precision / recall). AUROC and AUPRC are computed from the continuous `step_scores` and are invariant under `calib_mode`.

Data splits (per task):

```text
success_rollout            BCE train positives + calibration (pre-done frames)
success_rollout-val        benchmark eval success           (pre-done frames)
fail_rollout-labeled       GT-labeled failure bank (D_e/D_o via failure timing)
fail_rollout-val-labeled   benchmark eval failures
```

The train and eval success splits are disjoint; the runner hard-asserts
`train_success ∩ eval_success == ∅` (and the fail bank is disjoint from the eval
failures by `video_id`).

Run (robosuite) — important parameters are set directly inside the script
(`MODEL_CKPT`, `TASKS`, the four splits, head/calibration knobs); edit them
there rather than relying on env overrides:

```bash
bash robosuite/discriminator/dyn_disc/scripts/run_bce_robosuite_benchmark.sh
```

Or invoke the Python runner directly:

```bash
python -m robosuite.discriminator.dyn_disc.robosuite_bce \
  --model-ckpt /abs/path/to/checkpoints/model_50.pth \
  --data-root data --tasks PickPlaceCereal \
  --fail-split fail_rollout-val-labeled \
  --success-split success_rollout-val \
  --success-train-split success_rollout \
  --fail-train-split fail_rollout-labeled \
  --train-max-success-per-task 50 \
  --epochs 20 --fail-bank-per-task 25 --calib-mode two_class_youden
```

Peripheral env knobs (still overridable; important ones are hardcoded in-script):

```bash
EPOCHS=20
LR=3e-4
WEIGHT_DECAY=1e-4
BATCH_SIZE=512
DELTA=10.0
SAVE_CKPT_DIR=/path/to/out/checkpoints
```

Outputs are written under
`checkpoints/dyn_disc/bce_eval_robosuite/run_<timestamp>_<TASKS>/`.

Hard invariant: `BCEBenchmarkDiscriminator.fit_on_benchmark(...)` trains and calibrates only. It does not call `bench.evaluate(...)` and does not compute AUROC during training. Evaluation happens later in `robosuite_bce.py`.

---

## Per-trajectory visualization

The BCE visualization entry renders, for a sampled set of trajectories, an MP4 (per-frame HUD + red border on predicted-failure frames) plus a multi-page PDF (per-trajectory score curve, calibrated threshold, GT failure segments, summary page).

Important parameters (`MODEL_CKPT`, `TASK`, `SPLIT`, `NUM_TRAJS`, the four splits,
head knobs) are set directly inside the script — edit them there:

```bash
bash robosuite/discriminator/dyn_disc/scripts/visualize_bce_robosuite.sh
```

`SPLIT` selects the eval pool: `fail_rollout`, `success_rollout`, or `both`
(default). With `SPLIT=both`, `NUM_TRAJS=5` renders **5 failure and 5 success**
trajectories (i.e. `NUM_TRAJS` per pool). Success trajectories are scored only on
their pre-done (`is_success==False`) prefix; post-done frames are padded with a
low failure score so they never trigger a prediction.

The script mirrors `run_bce_robosuite_benchmark.sh` for failure-bank
construction, the disjoint train success split, and BCE head hyperparameters.

The visualizer always reports the two-class Youden operating point computed from the same `(train success-calib, fail-suffix)` failure-score distributions used by `CALIB_MODE=two_class_youden` in the benchmark runner. To inspect a different `tau`, change `CALIB_MODE` in the runner script and rerun — there is no separate viz knob.

By default the script fits a fresh BCE head and writes `bce_head.pth` under `<OUT_DIR>/checkpoints/`. To skip training and reuse a previously fitted head, pass `LOAD_CKPT=/abs/path/to/bce_head.pth` (the per-task thresholds are restored from the ckpt; failure-bank discovery is skipped).

Score / threshold semantics on screen:

```text
g(z)            = head(z)                # expert-likeness logit
bce_score       = -g(z)                  # shown in HUD; larger = more failure
pred_t = 1      iff bce_score_t >= tau_task
```

Output layout:

```text
<OUT_DIR>/
  videos/<video_id>.mp4
  bce_scores.pdf
  checkpoints/bce_head.pth    # only on cold fit; absent under LOAD_CKPT
```

---

## Public Python API

```python
from robosuite.discriminator.dyn_disc import (
    BCEDiscriminator,
    BCEBenchmarkDiscriminator,
    BCEHead,
    DetectionResult,
    DynBenchmarkDiscriminator,
    DynEncoder,
    load_model,
)
```

Use `BCEBenchmarkDiscriminator` when working with `benchmark.core.BenchmarkTrajectory`. Use `DynEncoder` or `BCEDiscriminator` directly for custom data loaders or online scoring.

Recommended explicit imports for new code:

```python
from robosuite.discriminator.dyn_disc.adapters import (
    BCEBenchmarkDiscriminator,
    DynBenchmarkDiscriminator,
)
from robosuite.discriminator.dyn_disc.detectors import (
    BCEDiscriminator,
    DynEncoder,
)
```

---

## Tests

BCE unit tests:

```bash
/home/dodo/miniconda3/envs/dagger/bin/python -m pytest \
  robosuite/discriminator/dyn_disc/tests/test_bce_discriminator.py -v
```

The tests cover head shape, synthetic separability, deterministic thresholding, scoring output, disjointness checks, class-balance capping, and state-dict roundtrip.

---

## Implementation notes

- Keep checkpoint tensors, `hydra.yaml`, and `normalizer.pth` together. `DynEncoder` depends on that layout.
- `camera_to_view` maps real camera names to training view names, for example `cam_high:agentview`.
- Larger `delta` lowers the calibration percentile and usually makes detectors stricter.
- Failure-bank based methods require usable `first_gt_failure_frame()` annotations.
- The benchmark adapters pad scores/predictions back to `trajectory.num_frames` when encoded feature length is shorter than the raw trajectory length.
