# Discriminator API

This document defines the standard offline discriminator API for this repository.

## Goal

Different discriminators may use different representations, losses, or calibration strategies, but evaluation and visualization code should interact with them through one stable interface.

## Base Class

Use:

- `robosuite.discriminator.utils.base.OfflineTrajectoryDiscriminator`

Every new discriminator should implement:

1. `name`
2. `fit(normal_bank_trajectories, calibration_trajectories=None)`
3. `detect_trajectory(trajectory, ...)`
4. `close()` if the implementation owns resources

## Required Semantics

### `fit(...)`

Purpose:

- build a normal reference bank,
- calibrate thresholds,
- or initialize any other state needed for inference.

Return type:

- `DetectorCalibrationSummary`

This summary should expose:

- detector name,
- initial threshold if one exists,
- metadata such as delta, checkpoint path, or configuration values.

### `detect_trajectory(...)`

Purpose:

- run one trajectory through the detector and return all step-wise outputs needed for:
  - metrics,
  - online threshold adaptation,
  - visualization.

Return type:

- `TrajectoryDetectionResult`

Required fields:

- `step_scores`: raw per-step scores before temporal aggregation
- `aggregate_scores`: temporal detector signal used for threshold comparison
- `thresholds`: threshold value used at each step
- `predictions`: binary per-step predictions

Optional fields:

- `aux_scores`
- `labels`
- `metadata`

## Design Rules

1. The detector should not hide the score used for final thresholding.
2. Evaluation code should not need to know model internals.
3. Visualization code should only depend on `TrajectoryDetectionResult`, not detector-specific tensors.
4. Comments and docstrings should stay concise and in English.

## Recommended Pattern

Each discriminator package should provide one wrapper class that implements the base API.

Example:

- `LPBKNNDiscriminator`
- `DynBCEDiscriminator`
- `FloatOfflineDiscriminator`

Then task-specific scripts can import the wrapper and call the shared evaluation / visualization helpers from:

- `robosuite.discriminator.utils.evaluation`
- `robosuite.discriminator.utils.visualization`
- `robosuite.discriminator.utils.video_io`
