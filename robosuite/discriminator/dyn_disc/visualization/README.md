# KNN Per-Trajectory Visualization

| File | Purpose |
| --- | --- |
| `visualize.py` | KNN discriminator visualization (`SingleBankVisualizer`). Supports both single-bank and two-bank KNN via `--mode {single_bank,two_bank}`. |
| `vis_robosuite_latent_dinov3.py` | Latent-space diagnostic visualization for the DINOv3 dynamics encoder. |

For a sampled set of failure trajectories, `visualize.py` renders:

- an MP4 per trajectory: per-frame HUD (`knn_dist`, `tau`, PRED/GT) plus a red
  border on predicted-failure frames;
- a multi-page PDF: a summary page followed by one per-trajectory score curve
  showing the score, the calibrated threshold `tau`, GT failure segments, and the
  first GT / first predicted failure frames.

## Modes

- `--mode single_bank` (default): fits a `SingleBankBenchmarkDiscriminator` on the
  per-task success eval set; the HUD score is the success-bank KNN min L2 distance.
- `--mode two_bank`: fits a `TwoBankBenchmarkDiscriminator`. The success eval set
  builds the success bank + calibration; a disjoint GT failure bank is discovered
  from `--fail-train-split` (default `fail_rollout-labeled`), sliced from
  `first_gt_failure_frame()` onward. The HUD score is the two-bank score
  (`difference` / `ratio` / `dsucc_only`). `video_id` disjointness between the eval
  failure set and the failure bank is hard-asserted.

## Threshold source

`--threshold-source` selects the `tau` used to draw borders / predicted-fail
frames:

- `detector` (default): the per-task calibrated threshold from `fit_on_benchmark`.
- `success_percentile`: a percentile of pooled success scores (`--step-success-percentile`).
- `fixed`: a constant `--fixed-threshold`.
- `--benchmark-json`: per-task trajectory best-F1 thresholds from a benchmark JSON.

## Entrances

```bash
# single-bank
MODEL_CKPT=/abs/path/to/checkpoint/model_50.pth TASK=PickPlaceCereal \
  bash robosuite/discriminator/dyn_disc/scripts/visualize_single_bank_robosuite.sh

# two-bank
MODEL_CKPT=/abs/path/to/checkpoint/model_50.pth TASK=PickPlaceCereal \
  bash robosuite/discriminator/dyn_disc/scripts/visualize_two_bank_robosuite.sh
```

Output layout:

```text
<OUT_DIR>/
  videos/<video_id>.mp4
  dyn_disc_scores.pdf
```
