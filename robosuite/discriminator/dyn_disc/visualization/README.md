# dyn_disc Per-Trajectory Visualization

Context and design notes for the discriminator visualization tool under
`dyn_disc/visualization/`. The dyn_disc top-level `../README.md` documents
user-facing invocation; this file captures the design rationale, interfaces, and
caveats so future edits can stay coherent.

## What lives here

| File | Purpose |
| --- | --- |
| `visualize_pu_bce.py` | nnPU (PU-BCE) discriminator visualization (`PUBCEVisualizer`). Robosuite only. Driven by `scripts/visualize_pu_bce_robosuite.sh`. |
| `vis_taco_latent.py` | CUDA-only TACO encoder inference followed by PCA and t-SNE. Driven by `scripts/vis_taco_latent_robosuite.sh`. |

It produces these artifacts per run:

```text
<out_dir>/
  videos/<video_id>.mp4        # per-frame HUD + red border on predicted-failure frames
  <pdf-name>.pdf               # 1 summary page + 1 page per sampled trajectory
  checkpoints/pu_bce_head.pth  # only on cold fit (absent under --load-ckpt)
```

## TACO latent-space visualization

The TACO latent visualizer uses `DynBenchmarkDiscriminator` only as the shared
trajectory preprocessing and encoder-cache layer. It does not fit an nnPU head
or run benchmark evaluation. Encoder inference is CUDA-only; PCA and t-SNE run
through NumPy / scikit-learn after the latent tensors have been copied to CPU.
Both latent and nnPU visualizations use the TACO `feature_source=encoder`
representation; this branch has no transformer-predictor feature mode.

```bash
MODEL_CKPT=/abs/path/to/model_50.pth TASK=NutAssemblyRound \
  bash robosuite/discriminator/dyn_disc/scripts/vis_taco_latent_robosuite.sh
```

The default output root is
`checkpoints/dyn_disc/ablations/TACO/visualizations/latent/`. Artifacts are
`taco_latent_pca.png`, `taco_latent_tsne.png`, `taco_latent_points.npz`, and
`taco_latent_points_meta.json`. The NPZ includes the original frame-level
encoder latent, both 2-D embeddings, sampled t-SNE indices, and phase labels:
`success`, `failure_before_gt`, and `failure_after_gt`.

## PU-BCE visualization — design

### Discriminator construction is shared with the runner

`PUBCEBenchmarkDiscriminator` inherits from `DynBenchmarkDiscriminator`, so its
`score_trajectory()` returns a `DiscriminatorOutput(step_scores, predictions,
aux)` identical in structure to any other head. Construction needs
`unlabeled_fail_trajectories` (the WHOLE failure pool — **no GT timing**) plus
the nnPU + head hyperparameters (`pi_p`, `loss_surrogate`, `nn_correction`,
`head_hidden`, ...).

Score semantics: `step_scores = -g(z)` (negative head logit, higher = more
failure-like). HUD text is `failure_score=-g(z)=...`.

The unlabeled-pool discovery / disjointness logic in
`visualize_pu_bce.py::_build_benchmark_and_pool` mirrors the runner
(`robosuite_pu_bce.py`): pull failure trajectories from `--fail-train-split`,
filter eval `video_id`s out for defence-in-depth, and use each failure
trajectory as a whole.

### Cold fit vs. `--load-ckpt`

Default behaviour calls `discriminator.fit_on_benchmark(eval_trajs)`, identical
to the benchmark runner; the trained head is saved to
`<out-dir>/checkpoints/pu_bce_head.pth`.

`--load-ckpt /path/to/pu_bce_head.pth` restores via `_bootstrap_from_ckpt(...)`:
unpacks `in_dim/hidden/num_layers`, head state dict, per-task thresholds,
calibration stats, then aliases the loaded detector into `disc._shared_detector`
and `disc._detectors_per_task[task]`. Skipping fit also skips pool discovery;
the discriminator is constructed with `unlabeled_fail_trajectories=[]`. The
disjointness invariant only runs inside `fit_on_benchmark`, so this is safe.

### Threshold: success_percentile only

The visualizer uses the detector's own per-task **success_percentile**
threshold: `tau = percentile(success-calib failure scores, 100 - delta)`. There
is no two-class Youden option in this branch — that rule needs failure labels,
which the PU formulation deliberately does not consume. To move the operating
point, change `--delta` and re-fit.

### Video orientation

Frames are flipped along the vertical axis by default (`canvas[:, ::-1, :, :]`)
to match the in-training image convention. Set `NO_FLIP_VERTICAL=1` if a future
dataset stores frames in the rendered orientation.

## File map

```text
visualize_pu_bce.py
├── argparse                     # encoder + nnPU knobs (robosuite-only)
├── _build_benchmark_and_pool    # benchmark factory + disjoint unlabeled pool
├── _bootstrap_from_ckpt         # restore fitted head without fit_on_benchmark
├── PUBCEVisualizer
│   ├── _score_trajectory        # uses PUBCEBenchmarkDiscriminator.score_trajectory()
│   ├── render_video             # ffmpeg/libx264 writer + HUD + border
│   └── render_pdf               # PdfPages: summary + per-trajectory panel
└── main                         # cold-fit OR --load-ckpt → sample → render
```
