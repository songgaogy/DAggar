# dyn_disc Visualization Library

`visualize_pu_bce.py` provides reusable nnPU rendering support for pipeline
visualization. It is intentionally a library module: it does not discover
training pools, fit a discriminator, run a benchmark, or expose a standalone
CLI.

The retained interfaces are:

- `PUBCEVisualizer` and `PerTrajectoryViz` for score rendering;
- video/PDF/HUD helpers used by the offline pipeline renderer;
- `_sample_trajectories_for_viz` for deterministic eval sampling;
- `_bootstrap_from_ckpt` for restoring a fitted head and calibrated thresholds;
- `_parse_camera_to_view` for shared camera mapping syntax.

Pipeline-owned entry points construct the discriminator, validate their data
contract, and pass fitted or loaded state into these helpers.

## Outputs

`PUBCEVisualizer.visualize(...)` writes:

```text
<out_dir>/
  videos/<video_id>.mp4
  <pdf-name>.pdf
```

For `split="both"`, videos are grouped below `videos/fail_rollout/` and
`videos/success_rollout/`. Each video includes a per-frame failure score,
calibrated task threshold, prediction, and red failure border. The PDF contains
a run summary and one score plot per sampled trajectory.

## Score and threshold semantics

```text
failure_score = -head_logit
pred_fail = failure_score >= tau_task
tau_task = percentile(success_calibration_failure_scores, 100 - delta)
```

The nnPU default is `delta=5`, so newly trained checkpoints use the P95 of
held-out success-calibration frame scores. Load-only visualization always uses
the threshold and delta stored in the checkpoint; legacy `delta=10` checkpoints
therefore retain their original P90 calibration.

Ground-truth failure segments may be displayed for inspection, but they are not
used to fit the nnPU head or calibrate the threshold.

Frames are vertically flipped by default to match the dynamics encoder image
convention. Callers can disable this through `PUBCEVisualizer(flip_vertical=False)`
when the source frames are already in display orientation.

The production head is trained by the single benchmark runner with the
`cap_c5_l1e2` epoch-1 default and a 20-epoch scheduler horizon. Visualization
only consumes its resulting checkpoint; it does not reproduce that training.
