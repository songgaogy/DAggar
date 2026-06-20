# dyn_disc Per-Trajectory Visualization

Context and design notes for the discriminator visualization tool under `dyn_disc/visualization/`. The dyn_disc top-level `../README.md` documents user-facing invocation; this file captures the design rationale, interfaces, and caveats so future edits can stay coherent.

## What lives here

| File | Purpose |
| --- | --- |
| `visualize_bce.py` | BCE discriminator visualization (`BCEVisualizer`) on the robosuite benchmark. Driven by `scripts/visualize_bce_robosuite.sh`. |

It produces these artifacts per run:

```text
<out_dir>/
  videos/<video_id>.mp4     # per-frame HUD + red border on predicted-failure frames
  <pdf-name>.pdf            # 1 summary page + 1 page per sampled trajectory
  checkpoints/bce_head.pth  # only on cold fit (absent under --load-ckpt)
```

## BCE visualization — design

### Discriminator construction is shared with the runner

`BCEBenchmarkDiscriminator` inherits from `DynBenchmarkDiscriminator` (the shared
encoding + feature-cache base), so its `score_trajectory()` returns a
`DiscriminatorOutput(step_scores, predictions, aux)`. The score semantics are:

- `step_scores = -g(z)` (negative head logit, higher = more failure-like). HUD text is `bce_score=...`.
- The PDF summary lists BCE-specific knobs (`head_hidden`, `head_layers`, `epochs`, `lr`, `max_expert_other_ratio`, `fail_bank_per_task`).

The bank-pool discovery / disjointness logic in `visualize_bce.py::_build_benchmark_and_bank` mirrors the runner (`robosuite_bce.py`): pull GT-labeled failures from a separate `--fail-train-split`, then filter eval `video_id`s out for defence-in-depth.

### Cold fit vs. `--load-ckpt`

Default behaviour calls `discriminator.fit_on_benchmark(eval_trajs)`, identical to the benchmark runner; the trained head is saved to `<out-dir>/checkpoints/bce_head.pth`.

`--load-ckpt /path/to/bce_head.pth` restores via `_bootstrap_from_ckpt(...)`:

1. `torch.load(payload)` → unpacks `in_dim`, `hidden`, `num_layers`, head state dict, per-task thresholds, calibration stats.
2. Constructs a `BCEDiscriminator` with those dimensions and calls `load_state_dict`.
3. Aliases the loaded detector into `disc._shared_detector` and `disc._detectors_per_task[task]` for every task present in the saved thresholds table.
4. Populates `disc._global_stats` and `disc._calibration_stats` from the payload so `calibration_summary()` still drives the PDF summary page.

Skipping fit also skips fail-bank discovery; the discriminator is constructed with `fail_bank_trajectories=[]`. The disjointness invariant in the adapter only runs inside `fit_on_benchmark`, so this is safe.

### Threshold: always two-class Youden

The visualizer always reports the two-class Youden operating point. For the
sampled task the visualizer computes:

```text
tau = argmax_t  TPR(t) - FPR(t)
```

where `TPR` is over fail-bank GT-suffix frames (`[first_gt_failure_frame:]`)
and `FPR` is over the success-calib slice that the adapter used during fit.

Implementation notes:

- The two empirical samples are scored by re-running the fitted head over the
  fail-bank trajectories and over the success-calib trajectories (re-derived
  here with the same `(seed, calib_fraction)` rule as
  `BCEBenchmarkDiscriminator.fit_on_benchmark`).
- The shared `two_class_youden_threshold` helper lives in
  `detectors/bce.py` and is also used by the benchmark runner's
  `calib_mode=two_class_youden`. Visualizer + runner therefore agree on the
  same `tau` definition.
- The split-replication logic in `_compute_success_calib_failure_scores_per_task`
  must stay in sync with the adapter's fit; changes to either side will mix
  train + calib frames into the success sample. This is the only fragile
  coupling — single helper, easy to keep aligned.
- Works under both cold fit and `--load-ckpt`. Under `--load-ckpt` the visualizer
  re-discovers the bank pool from `--fail-train-split` purely for scoring (the
  head and weights still come from disk).

To inspect a different operating point, change `CALIB_MODE` in the benchmark
runner (`run_bce_robosuite_benchmark.sh`) and re-fit. The visualizer does not need a
flag — its tau is reproducible from the eval set + bank pool alone.

### Rendering helpers

`visualize_bce.py` defines its own `_draw_border`, `_pad_to_even`, `_load_font`, and `_percentile_summary` helpers (self-contained, no cross-module imports).

### Video orientation

Frames are flipped along the vertical axis by default (`canvas[:, ::-1, :, :]`) to match the in-training image convention. Set `NO_FLIP_VERTICAL=1` if a future dataset stores frames in the rendered orientation.

## Knob naming

`--feature-source` and `--transformer-layer` are encoder-level knobs; the BCE visualizer exposes them without the `--knn-` prefix (BCE doesn't run a KNN search at score time). The corresponding bash env vars are `FEATURE_SOURCE` and `TRANSFORMER_LAYER`. The benchmark runner still uses the `--knn-feature-source` / `--knn-transformer-layer` historical names; that is intentional for backwards compatibility and not coupled to this tool.

## Verification path

Cold fit + viz, then warm reload of the produced ckpt:

```bash
# 1. Cold robosuite run
MODEL_CKPT=checkpoints/dyn_disc/dynamics/<run>/checkpoint/model_50.pth \
TASK=PickPlaceCereal NUM_TRAJS=3 \
  bash robosuite/discriminator/dyn_disc/scripts/visualize_bce_robosuite.sh

# 2. Warm reload (skip training)
MODEL_CKPT=... \
LOAD_CKPT=<OUT_DIR_FROM_STEP_1>/checkpoints/bce_head.pth \
TASK=PickPlaceMilk NUM_TRAJS=2 \
  bash robosuite/discriminator/dyn_disc/scripts/visualize_bce_robosuite.sh
```

Sanity checks per run:

- `videos/*.mp4` count equals `NUM_TRAJS`.
- Red border appears in or near GT failure frames; HUD `PRED:` colour matches.
- PDF summary page lists BCE-specific hyperparameters.
- Per-trajectory PDF page shows the score curve crossing `tau` consistently with the binary `predictions`.

## File map

```text
visualize_bce.py
├── argparse                   # BCE / encoder knobs
├── _build_benchmark_and_bank  # robosuite benchmark factory + disjoint bank pool
├── _bootstrap_from_ckpt       # restore fitted head without fit_on_benchmark
├── BCEVisualizer
│   ├── _score_trajectory      # uses BCEBenchmarkDiscriminator.score_trajectory()
│   ├── render_video           # ffmpeg/libx264 writer + HUD + border
│   └── render_pdf             # PdfPages: summary + per-trajectory panel
└── main                       # cold-fit OR --load-ckpt → sample → render
```
