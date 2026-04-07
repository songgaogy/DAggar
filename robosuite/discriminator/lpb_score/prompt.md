# LPB Score Three-Headed DSM

This module implements an offline failure detector based on a **Three-Headed Conditional Denoising Score Matching (DSM)** model over latent transitions.

## Goal

Given a transition

`(z_t, a_{t:t+H-1}, z_{t+H})`

the detector estimates how far this transition is from the expert manifold. The latent `z_t` is produced by the frozen flow policy encoder and is not changed by this module.

The detector is designed to answer three questions at every step:

1. Is the current latent state itself unusual?
2. Is the action sequence unusual given the current state?
3. Is the realized transition unusual given the current state and action?

## Conditional Factorization

Instead of reconstructing the entire transition tuple with a single head, the model uses the causal factorization

`p(z_t, a, z_{t+H}) = p(z_t) p(a | z_t) p(z_{t+H} | z_t, a)`

and matches it with three denoising heads:

1. **State Head**
   Input: noisy normalized `z_t`
   Target: clean normalized `z_t`

2. **Actor Head**
   Input: clean normalized `z_t` and noisy normalized action chunk
   Target: clean normalized action chunk

3. **Dynamics Head**
   Input: clean normalized `z_t`, clean normalized action chunk, and noisy normalized `z_{t+H}`
   Target: normalized residual

The dynamics head does **not** predict absolute next state anymore. It predicts the residual

`delta_z = z_{t+H}^{norm} - z_t^{norm}`

which preserves local transition geometry and gives sharper dynamics sensitivity.

## Normalization

Before denoising, the latent and action features are standardized using dataset statistics computed from expert trajectories:

- `latent_mean`
- `latent_var`
- `action_mean`
- `action_var`

The corresponding standard deviations are clamped:

`std = clamp(sqrt(var), min=std_clamp_min)`

with default

`std_clamp_min = 0.05`

This prevents near-constant features from exploding during normalization.

## Training Objective

File: `core/model.py`

Training happens in normalized space.

Let

- `z_t_norm` be the normalized current latent
- `a_norm` be the normalized flattened action chunk
- `z_next_norm` be the normalized future latent
- `delta_z_norm = z_next_norm - z_t_norm`

Gaussian noise is injected **after** normalization:

- `z_noisy = z_t_norm + eps_z`
- `a_noisy = a_norm + eps_a`
- `z_next_noisy = z_next_norm + eps_next`

with default

`noise_scale = 0.08`

The three heads are trained with per-dimension mean squared error:

- `state_energy = mean((state_hat - z_t_norm)^2)`
- `policy_energy_raw = mean((action_hat - a_norm)^2)`
- `dynamics_energy = mean((delta_hat - delta_z_norm)^2)`

The training loss is

`loss = state_energy + policy_energy_raw + dynamics_energy`

Important implementation choices:

- all Dropout layers were removed to avoid mean-collapse
- the dynamics head predicts residual `delta_z_norm`
- reduction across feature dimensions uses `mean(dim=-1)`, not `sum(dim=-1)`, so state and action heads are normalized by their own degrees of freedom

## Inference Score

File: `core/dsm_discriminator.py`

At test time, the model runs with

`add_noise = False`

so no Gaussian perturbation is added during scoring.

For each timestep, the model computes:

- `state_energy`
- `policy_energy_raw`
- `dynamics_energy`

To reflect the fact that expert action behavior is naturally more stochastic than physical state and dynamics, the policy term is temperature-scaled by a dedicated factor:

`policy_energy = policy_weight * policy_energy_raw`

with default

`policy_weight = 0.2`

The final step score is

`u_t = state_energy + policy_energy + dynamics_energy`

This keeps degree-of-freedom normalization from `mean(dim=-1)` while softly suppressing false positives caused by natural action jitter.

## Temporal Aggregation

The per-step score `u_t` is smoothed into `lambda_t` using the existing temporal aggregation logic:

- `lambda_mode = mean` or `max`
- `lambda_window_size` controls prefix or rolling aggregation

Thresholds are then calibrated from clean validation trajectories using the configured `delta` percentile rule.

## Visualization Semantics

Files:

- `core/dsm_discriminator.py`
- `visualize_failures.py`
- `app/visualize.py`

The visualization pipeline uses the already-scaled component energies exported by the discriminator:

- `state_error -> state_energy`
- `action_error -> scaled policy_energy`
- `next_state_error -> dynamics_energy`

This means the contribution shares and stacked area plots are consistent with the actual detection score:

`u_t = state_energy + scaled_policy_energy + dynamics_energy`

The policy attribution shown in plots is therefore intentionally temperature-reduced rather than raw.

## Main Hyperparameters

Relevant tuning knobs:

- `std_clamp_min`
  Lower bound for normalization standard deviation
  Default: `0.05`

- `noise_scale`
  Gaussian noise scale used during training after normalization
  Default: `0.08`

- `policy_weight`
  Temperature on the policy head during inference and visualization
  Default: `0.2`

- `lambda_mode`
  Temporal aggregation mode for `u_t`

- `lambda_window_size`
  Window size for rolling aggregation

- `delta`
  Quantile parameter for threshold calibration

## Practical Interpretation

- `state_energy` rises when the visual / latent state itself becomes unfamiliar
- `policy_energy` rises when the chosen action is atypical for the current latent context
- `dynamics_energy` rises when the realized transition violates learned physical transition structure

For failures like object drops or severe scene deviations, the most informative terms are usually `state_energy` and `dynamics_energy`, while `policy_energy` is treated as a softer behavioral prior.

## Implementation Notes

- the frozen flow policy encoder is unchanged
- KNN-based LPB metrics are no longer used in the DSM path
- policy checkpoints and DSM checkpoints are now validated separately in the helper scripts
- visualization, evaluation, and analysis all consume the same scaled inference energies
