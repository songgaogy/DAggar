# Discriminator Visualization

The visualization tools reuse the frozen policy encoder and the same feature
cache as the benchmark runner. Policy tensor inference is CUDA-only.

## Tools

| File | Purpose |
| --- | --- |
| `visualize_pu_bce.py` | Render per-trajectory MP4 files and a score PDF from a fitted or newly trained nnPU head. |
| `vis_policy_latent.py` | Encode `task_scene_cond` features, then generate PCA and t-SNE plots plus machine-readable artifacts. |

Both launchers write below
`checkpoints/dyn_disc/ablations/policy/runs/run_<timestamp>_<task>/visualizations/`.

## Policy latent visualization

Run:

```bash
TASK=NutAssemblyRound \
  bash robosuite/discriminator/dyn_disc/scripts/vis_policy_latent_robosuite.sh
```

Useful overrides include `POLICY_CKPT`, `DATA_ROOT`, `OUT_DIR`,
`FEATURE_CACHE_DIR`, `ENCODE_BATCH_SIZE`, `PRELOAD_WORKERS`,
`PREFETCH_FACTOR`, `PIN_MEMORY`, `REUSE_FEATURE_CACHE`, and `SEED`. Additional
CLI arguments are forwarded by the launcher.

The output directory contains:

```text
policy_latent_pca.png
policy_latent_tsne.png
policy_latent_points.npz
policy_latent_points_meta.json
```

The NPZ stores the original frame-level 256D latent, PCA coordinates, sampled
t-SNE coordinates and indices, task/video/frame labels, and one of these phase
labels for every frame:

- `success`
- `failure_before_gt`
- `failure_after_gt`

The JSON records the policy checkpoint hash, EMA selection, camera order,
prompt map, preprocessing version, cache settings, dimensionality-reduction
seed, artifact paths, and per-trajectory failure metadata.

Ground-truth failure onset is read only after feature extraction for offline
plot coloring. It is never passed to the encoder, nnPU fit, calibration, or
benchmark evaluation. A failure trajectory without an onset annotation is
entirely labeled `failure_before_gt` in the plot.

## Video and score visualization

`visualize_pu_bce.py` renders an MP4 with a per-frame HUD and a red border on
predicted-failure frames, plus a multi-page PDF of score curves. Score semantics
are:

```text
g(z)          = success-likeness head logit
failure_score = -g(z)
pred_fail     = failure_score >= task threshold
```

By default, the visualizer fits a fresh head and saves
`checkpoints/pu_bce_head.pth` inside the run directory. Passing `--load-ckpt`
restores the head and per-task thresholds and skips fitting. Checkpoint metadata
must match the current policy checkpoint, EMA state, camera order, prompt, and
preprocessing contract.

The threshold is calibrated only from held-out success frames at the requested
false-alarm percentile. GT failure timing may be displayed in the PDF or video,
but does not affect fitting or threshold calibration.
