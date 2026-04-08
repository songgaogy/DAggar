# LPB Score Unified Task-Conditioned DSM

This module implements an offline failure detector based on a **Unified Task-Conditioned Diffusion Transformer (DiT)** over latent transitions.

## Goal

Given a transition

`(z_t, a_{t:t+H-1}, z_{t+H})`

the detector estimates how far the transition is from the expert manifold. The latent `z_t` comes from the frozen flow policy encoder and is not changed by this module.

The detector still evaluates three conditional factors at each step:

1. Is the current latent state itself unusual?
2. Is the action chunk unusual given the current state?
3. Is the realized transition unusual given the current state and action?

## Unified Routing

The model keeps the same causal factorization

`p(z_t, a, z_{t+H}) = p(z_t) p(a | z_t) p(z_{t+H} | z_t, a)`

but routes the three targets through one shared backbone with task IDs:

- `0`: state validation
- `1`: actor validation
- `2`: dynamics validation

For each task, the model builds a 3-token sequence:

1. task token from `nn.Embedding(3, embed_dim)`
2. context token from the clean condition vector
3. target token from the noisy target variable

The target and context vectors are zero-padded to shared maximum dimensions before projection:

- `target_max_dim = max(latent_dim, action_flat_dim)`
- `context_max_dim = latent_dim + action_flat_dim`

Task routing uses:

- state task: target=`z_t_norm`, context=`0`
- actor task: target=`a_norm`, context=`z_t_norm`
- dynamics task: target=`delta_z_norm`, context=`[z_t_norm, a_norm]`

The backbone is a shared `nn.TransformerEncoder` and only the output feature of the target token is decoded.

## Normalization

Before denoising, latent and action features are standardized with statistics computed from expert trajectories:

- `latent_mean`
- `latent_var`
- `action_mean`
- `action_var`

Standard deviations are clamped:

`std = clamp(sqrt(var), min=std_clamp_min)`

with default

`std_clamp_min = 0.05`

This keeps near-constant features stable during normalization.

## Training Objective

File: `core/model.py`

Training happens in normalized space.

Let

- `z_t_norm` be the normalized current latent
- `a_norm` be the normalized flattened action chunk
- `z_next_norm` be the normalized future latent
- `delta_z_norm = z_next_norm - z_t_norm`

Gaussian noise is injected after normalization:

- `z_noisy = z_t_norm + eps_z`
- `a_noisy = a_norm + eps_a`
- `delta_noisy = delta_z_norm + eps_delta`

with default

`noise_scale = 0.08`

Each task predicts:

- `pred_mean`
- `pred_logvar`

The per-dimension training objective is the heteroscedastic Gaussian NLL:

`0.5 * ((pred_mean - target)^2 * exp(-pred_logvar) + pred_logvar)`

Per-task energies use `mean(dim=-1)` over active feature dimensions. The final score is:

`u_t = state_energy + action_energy + dynamics_energy`

The optimization loss is the batch mean of `u_t`.

Diagnostic MSE metrics are still logged, but they no longer drive training.

## Inference Score

File: `core/dsm_discriminator.py`

At test time, the model runs with

`add_noise = False`

so inference uses the clean normalized state, clean normalized action chunk, and clean normalized dynamics residual.

For each timestep, the detector computes:

- `state_energy`
- `action_energy`
- `dynamics_energy`

Each term is the model NLL for that routed task. There is no manual `policy_weight` anymore. Action stochasticity is handled by the predicted `logvar`.

The final step score is:

`u_t = state_energy + action_energy + dynamics_energy`

## Temporal Aggregation

The per-step score `u_t` is aggregated into `lambda_t` using the existing temporal logic:

- `lambda_mode = mean` or `max`
- `lambda_window_size` controls prefix or rolling aggregation

Thresholds are calibrated from clean validation trajectories with the configured `delta` percentile rule.

## Visualization Semantics

Files:

- `core/dsm_discriminator.py`
- `visualize_failures.py`
- `app/visualize.py`

Visualization uses the detector component energies directly:

- `state_error -> state_energy`
- `action_error -> action_energy`
- `next_state_error -> dynamics_energy`

The stacked attribution plots therefore match the actual detection score:

`u_t = state_energy + action_energy + dynamics_energy`

## Main Hyperparameters

- `embed_dim`
  Transformer token width

- `num_layers`
  Number of transformer encoder layers

- `num_heads`
  Number of attention heads

- `ffn_dim`
  Transformer feed-forward width

- `std_clamp_min`
  Lower bound for normalization standard deviation

- `noise_scale`
  Gaussian noise scale used during training after normalization

- `lambda_mode`
  Temporal aggregation mode for `u_t`

- `lambda_window_size`
  Window size for rolling aggregation

- `delta`
  Quantile parameter for threshold calibration

## Practical Interpretation

- `state_energy` rises when the latent state itself becomes unfamiliar
- `action_energy` rises when the chosen action chunk is atypical for the current latent
- `dynamics_energy` rises when the realized transition violates learned transition structure

For failures like object drops or severe scene deviations, `state_energy` and `dynamics_energy` are often the sharpest indicators, while `action_energy` is automatically softened when the model predicts large uncertainty.

## Implementation Notes

- the frozen flow policy encoder is unchanged
- the dynamics task predicts `delta_z_norm`, not absolute next latent
- old three-head DSM checkpoints are intentionally incompatible with the unified model
- evaluation, analysis, and visualization all consume the same NLL-based component energies
