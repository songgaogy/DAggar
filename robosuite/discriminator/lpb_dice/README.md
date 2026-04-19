# LPB Dice: A PU-Calibrated Latent Failure Detector

`lpb_dice` is an offline failure detector on latent transitions from a frozen policy encoder.
It is designed for settings where:

- positive trajectories are available (`expert + success_rollout`),
- fail trajectories are available but treated as **contaminated background** (not clean negatives),
- a calibrated sequential decision score is needed for per-step failure detection.

## 1) Problem Setup

For each trajectory, let:

- `z_t` be the latent state at step `t` from a frozen policy encoder,
- `a_{t:t+H-1}` be an action chunk of horizon `H`,
- `z_{t+H}` be the future latent.

The detector learns a scalar score indicating whether a transition is likely to be outside the positive manifold.

Data roles in this module:

- **Positive (`P`)**: `expert`, `success_rollout`
- **Background (`B`)**: `fail_rollout` (may include unknown positive fraction)

This is a Positive-Unlabeled / Positive-Background style setup.

## 2) Transition Representation

### 2.1 Frozen world-model feature

A latent dynamics model provides a transition feature:

`h_t = f_theta(z_t, a_{t:t+H-1})`

where `f_theta` is frozen during PU head training.

The detector input is the concatenation:

`x_t = [z_t ; h_t]`

This is implemented in `core/representation.py` (`FrozenTransitionRepresentation`).

### 2.2 Support penalty raw term

The same frozen dynamics model predicts `hat{z}_{t+H}` (and optionally log-variance).
A per-step transition inconsistency is computed as:

- with uncertainty head:
  `r_t = 0.5 * mean_i( (e_{t,i}^2 * exp(-logvar_{t,i}) + logvar_{t,i}) )`
- without uncertainty head:
  `r_t = mean_i(e_{t,i}^2)`

where `e_t = hat{z}_{t+H} - z_{t+H}`.

This raw penalty is later normalized and clipped to non-negative values.

## 3) PU Classifier Head

A small MLP classifier `g_phi(x_t)` is trained with BCE-with-logits:

`l_t = g_phi(x_t),    p_t = sigma(l_t)`

Labels are:

- `y_t = 1` for positive transitions,
- `y_t = 0` for background transitions.

Training objective:

`L_BCE = E[ -y_t log p_t - (1-y_t) log(1-p_t) ]`

The loader uses weighted sampling to enforce a configured positive ratio in each epoch.

Implementation: `core/pu_detector.py` (`train_pu_classifier`).

## 4) PU Calibration

Raw classifier probabilities are corrected with a class-prior-like constant estimated from positive calibration data.

### 4.1 Estimating `c`

On positive calibration transitions:

`c = max( mean(p_t), c_min )`

Intuition: for a non-traditional classifier over positive/background data, dividing by `c` rescales confidence toward a positive-posterior proxy.

### 4.2 Corrected probability and logit score

`tilde{p}_t = clip(p_t / c, 0, 1)`

`s_t^{pu} = logit( clip(tilde{p}_t, eps, 1-eps) )`

The detector then uses the negative of this term as anomaly evidence (larger means more failure-like):

`a_t^{pu} = - s_t^{pu}`

Implementation: `calibrate_pu_detector`, `detect_trajectory` in `core/pu_detector.py`.

## 5) Support Penalty Normalization

Raw support penalties from positive calibration trajectories are standardized:

`u_t = max( (r_t - mu_r) / sigma_r, 0 )`

where `mu_r, sigma_r` are mean/std over calibration positives.

Final step score:

`q_t = a_t^{pu} + w_support * u_t`

Default in current config: `w_support = 0.0`.

Implementation: `core/support.py`.

## 6) Temporal Aggregation and Threshold

Per-step scores `q_t` are aggregated into a running statistic `lambda_t`:

- `lambda_mode = mean`: rolling/prefix mean
- `lambda_mode = max`: rolling/prefix max

with window size `W = lambda_window_size` (`W <= 0` means full prefix).

Threshold is calibrated on positive trajectories:

`tau = Quantile_{1 - delta/100}( {lambda_t on positive calibration} )`

(`delta` is in percent, e.g. `delta=10` gives the 90th percentile).

Prediction rule:

`hat{y}_t = 1[ lambda_t >= tau ]`

and `first_crossing_index` is the first `t` with `hat{y}_t = 1`.

## 7) End-to-End Training Pipeline

Implemented in `app/train.py`:

1. Build latent cache from raw trajectories.
2. Load or train latent dynamics backbone.
3. Freeze backbone; encode each transition into `x_t` and `r_t`.
4. Train PU head on train split.
5. Calibrate `c`, support normalization, and threshold on eval calibration splits.
6. Save a combined checkpoint (`format_version = lpb_dice_v1`) containing:
   - backbone payload,
   - PU head state/config,
   - calibration state.

## 8) Inference and Visualization

`LPBDiceDiscriminator` loads one combined checkpoint and outputs per trajectory:

- `pu_raw_score` (logits),
- `pu_corrected_score` (corrected logit),
- `support_penalty` (normalized),
- `final_step_score = q_t`,
- `aggregate_scores = lambda_t`,
- `thresholds = tau`.

Visualization entrypoint (`visualize_failures.py`) renders MP4/PDF and stores `summary.json`.

## 9) Entrypoints

### Train

- Python: `robosuite/discriminator/lpb_dice/train.py`
- Bash: `robosuite/discriminator/lpb_dice/scripts/train_lpb_dice.sh`

Example:

```bash
bash robosuite/discriminator/lpb_dice/scripts/train_lpb_dice.sh
```

### Visualize

- Python: `robosuite/discriminator/lpb_dice/visualize_failures.py`
- Bash: `robosuite/discriminator/lpb_dice/scripts/visualize_lpb_dice_failures.sh`

Example:

```bash
bash robosuite/discriminator/lpb_dice/scripts/visualize_lpb_dice_failures.sh
```

## 10) Key Assumptions and Practical Notes

- `fail_rollout` is used as **background**, not guaranteed clean negative.
- Calibration quality depends on the representativeness of positive calibration trajectories.
- `w_support > 0` adds transition-model inconsistency prior; `w_support = 0` gives pure PU-based scoring.
- Threshold is computed from positive calibration only, so `delta` directly controls tolerance to false alarms on positive data.
