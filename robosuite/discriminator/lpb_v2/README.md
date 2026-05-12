# LPB v2 — Latent dynamics pretraining + KNN discriminator

This package is a **self-contained** implementation of an **LPB-style visual dynamics model** (multi-view ResNet encoders, proprio/action MLP encoders, ViT predictor) trained with **latent prediction / reconstruction** on demonstration trajectories, followed by a **simple k-nearest-neighbor (KNN) out-of-distribution (OOD) discriminator** on top of frozen (or partially trained) features.

Conceptually it follows a **world-model–style pretraining** recipe: learn a compact transition representation by predicting future latent tokens from short history, then use that latent space for **non-parametric** scoring at test time (minimum L2 distance to an in-distribution memory bank, with a percentile-calibrated threshold). This is aligned with common “pretrain encoder / latent dynamics, then cheap downstream probe” pipelines (e.g. dynamics-style self-supervision similar in spirit to WAM-style latent learning, without tying this repo to a single external paper acronym).

---

## What is implemented

| Stage | Role |
|--------|------|
| **Pretraining** | `VisualDynamicsModel`: encode `(images, proprio, action)` into tokens, run a temporal ViT predictor, minimize MSE between predicted and target latents (visual / proprio / optional action terms). |
| **Latent / features for scoring** | `LPBV2Encoder`: loads a checkpoint + `hydra.yaml` + `normalizer.pth`, applies the same normalization and crops as training, emits either **concatenated encoder embeddings** (`encoder`) or **transformer hidden states** (`transformer`). |
| **Discrimination** | `LPBV2KNN`: build a **feature bank** from expert success demos, calibrate a threshold on a **disjoint calibration** split, score new frames by **min L2 distance** to the bank (with fixed per-block weights on visual / proprio / action dimensions). Binary “failure” when distance ≥ threshold. |
| **Benchmark glue** | `LPBV2BenchmarkDiscriminator` implements the shared `benchmark` trajectory API (`fit_on_benchmark`, `score_trajectory`). |

Training is **single-GPU**, **Hydra-configured**, and intentionally **lightweight** (no Accelerate), while keeping checkpoint layout compatible with `model_loader.load_model`.

---

## Repository layout

```text
lpb_v2/
  train.py                 # Hydra entry: dynamics pretraining
  knn.py                   # Encoder wrapper + KNN + distance helpers
  benchmark.py             # FailureBenchmark adapter
  run_benchmark.py         # CLI over FailureBenchmark + LPBV2
  visualize.py             # Optional visualization utilities
  model_loader.py          # Instantiate / load checkpoints
  config/                  # Hydra defaults (env, encoders, predictor, train)
  data/                    # HDF5 and preprocessed-cache datasets
  models/                  # ResNet, ViT predictor, VisualDynamicsModel, …
  utils/                   # Normalizer, tensor helpers, latent vis
  scripts/                 # Bash wrappers (train, benchmark, vis, …)
```

---

## Training (latent dynamics)

**Entry point:** `python -m robosuite.discriminator.lpb_v2.train` with Hydra (`config/train.yaml`).

**Data backends (pick via `env=`):**

- **`env=hdf5`** — `HDF5DynamicsModelDataset` on robosuite-style rollout HDF5 trees.
- **`env=preprocessed`** — `PreprocessedCacheDynamicsModelDataset` on pre-exported `.npz` caches (used by the provided training script).

**Outputs under the Hydra run directory** (default pattern in `config/train.yaml`):

- `hydra.yaml` — resolved config (required for loading).
- `normalizer.pth` — image + state + action normalization (required for KNN encoding to match training).
- `checkpoints/model_<epoch>.pth` — weights for `encoder`, `predictor`, `proprio_encoder`, `action_encoder`, plus optional LayerNorm buffers.

**Convenience script (from repo root):**

```bash
# Example: single task; see script for env vars (CACHE_ROOT, TASKS, epochs, …)
TASK=PickPlaceCan bash robosuite/discriminator/lpb_v2/scripts/train_lpb_v2_dynamics.sh
```

Override training knobs via Hydra CLI as usual, e.g. `training.epochs=20`, `frameskip=4`, `model.train_encoder=true`.

---

## KNN discriminator

**Core types** (`knn.py`):

- `LPBV2Encoder` — resolves `hydra.yaml` / `normalizer.pth` next to the checkpoint, loads the dynamics model, exposes `encode_batch(...)`.
- `LPBV2KNN` — `fit(expert_features, calibration_features)` then `score(features)`.

**Feature sources:**

- `feature_source="encoder"` — KNN on **`[visual_emb ; proprio_emb ; action_emb]`** with separate `visual_weight`, `proprio_weight`, `action_weight`.
- `feature_source="transformer"` — KNN on a **flattened ViT hidden state** at `transformer_layer` (block weights collapse to uniform L2 in the benchmark adapter).

**Threshold `delta`:** percentile on calibration min-distances; larger `delta` tends to make the detector **stricter** (lower percentile → lower threshold → more frames flagged). See docstrings in `knn.py` for the exact mapping.

**Aliases:** `LPBOriginalEncoder` / `LPBOriginalKNN` point to the v2 classes for backward compatibility with older naming.

---

## Running the failure benchmark

**CLI:** `python -m robosuite.discriminator.lpb_v2.run_benchmark`  
(requires the shared `benchmark` package and rollout roots used elsewhere in this repo.)

**Example script:**

```bash
MODEL_CKPT=/abs/path/to/.../checkpoints/model_49.pth \
  bash robosuite/discriminator/lpb_v2/scripts/run_lpb_v2_benchmark.sh
```

The script wires `FailureBenchmark` + `LPBV2BenchmarkDiscriminator` and writes JSON (path controlled by `SAVE_JSON` / `OUT_DIR`). Adjust `KNN_FEATURE_SOURCE`, `KNN_TRANSFORMER_LAYER`, weights, and `DELTA` via environment variables documented in the shell file.

---

## Public Python API

```python
from robosuite.discriminator.lpb_v2 import (
    LPBV2BenchmarkDiscriminator,
    LPBV2Encoder,
    LPBV2KNN,
    load_model,
)
```

Use `LPBV2BenchmarkDiscriminator` when integrating with `benchmark.core.BenchmarkTrajectory`; use `LPBV2Encoder` / `LPBV2KNN` directly for custom datasets or online scoring.

---

## Dependencies & environment

- **PyTorch**, **Hydra**, **OmegaConf**, **torchvision** (ResNet), **einops** (model internals), and project modules (`benchmark`, robosuite data paths as configured).
- In this workspace, training is typically run with the **`daggar`** conda env (see repo `AGENTS.md`) or `PYTHON_BIN` in the bash scripts.

---

## Related code

- **`robosuite.discriminator.lpb_original`** — legacy naming / datasets referenced in configs; v2 maps some dataset class paths when rebuilding old configs.
- **`visualize.py`** / `scripts/visualize_lpb_v2.sh` — latent visualization helpers.

If you extend encoders, action layout, or cache format, keep **`hydra.yaml`**, **`normalizer.pth`**, and checkpoint tensors **consistent** so `LPBV2Encoder` stays bitwise aligned with training.
