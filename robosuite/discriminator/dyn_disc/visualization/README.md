# RPT and nnPU Visualizations

This directory contains two robosuite-only visualization entry points.

| Module | Outputs |
| --- | --- |
| `visualize_pu_bce.py` | nnPU score MP4s plus a multi-page PDF |
| `visualize_rpt_latents.py` | PCA/t-SNE PNGs, raw NPZ data, and metadata JSON |

## nnPU video visualization

`PUBCEVisualizer` calls the benchmark adapter's public
`score_trajectory()` interface. Each video contains the failure score
`-g(z)`, the calibrated per-task threshold, and a red frame border when the
detector fires. The PDF includes calibration metadata and one score curve per
trajectory.

`--load-ckpt` is the standard visualization path. It loads the saved nnPU head
and thresholds, verifies `pretraining_method=rpt`, and checks that the head's
`representation_fingerprint` equals the loaded RPT checkpoint fingerprint.
This path only builds the evaluation benchmark and deliberately skips
unlabeled-pool discovery. Cold fit remains available for debugging and uses the
same whole-trajectory U semantics as nnPU training.

Frames are vertically flipped by default to retain the existing visualization
convention. Use `--no-flip-vertical` or launcher environment variable
`NO_FLIP_VERTICAL=1` to disable it.

Output layout:

```text
<out-dir>/
  videos/[fail_rollout|success_rollout]/<video-id>.mp4
  pu_bce_scores.pdf
```

## RPT latent visualization

`visualize_rpt_latents.py` uses
`DynBenchmarkDiscriminator.preload_trajectory()` and
`encode_preloaded()` so plots use exactly the same causal RPT action-token
features as nnPU. It samples success and labeled failure evaluation
trajectories, then labels frames as:

- `success`
- `failure-before-GT`
- `failure-after-GT`

The GT mask is used only for plot coloring. Encoding is CUDA-only; PCA and
t-SNE run in scikit-learn after features have been copied to NumPy.

Output layout:

```text
<out-dir>/
  pca.png
  tsne.png
  rpt_latents.npz
  metadata.json
```

The NPZ stores the 192-D features, integer labels and names, both embeddings,
video IDs, and frame indices. `--max-points` deterministically subsamples before
PCA/t-SNE to bound runtime and memory.
