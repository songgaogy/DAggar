# LPB Score DSM Implementation Summary

This module now uses a **Three-Headed Conditional Denoising Score Matching (DSM)** detector for offline failure detection on latent transitions.

## Overview

Given a transition tuple

`(z_t, a_{t:t+H-1}, z_{t+H})`

the model does not denoise the whole tuple with one joint head. Instead, it factorizes the conditional reconstruction into three heads:

1. **State head**
   Input: noisy `z_t`
   Target: clean `z_t`

2. **Actor head**
   Input: clean `z_t` and noisy action chunk `a_{t:t+H-1}`
   Target: clean action chunk `a_{t:t+H-1}`

3. **Dynamics head**
   Input: clean `z_t`, clean action chunk `a_{t:t+H-1}`, and noisy `z_{t+H}`
   Target: clean `z_{t+H}`

This matches the intended causal factorization:

`p(z_t, a, z_{t+H}) = p(z_t) p(a | z_t) p(z_{t+H} | z_t, a)`

## Model Implementation

File: `core/model.py`

- `ConditionalManifoldDenoiser` contains three separate towers:
  - `state_tower`
  - `actor_tower`
  - `dynamics_tower`
- `DSMModel.forward(...)` builds clean inputs, optionally adds Gaussian noise, and routes each branch with the correct conditioning:
  - state branch uses `state_input = z_t + eps_z`
  - actor branch uses clean `z_t` and `action_input = a + eps_a`
  - dynamics branch uses clean `z_t`, clean `a`, and `next_state_input = z_{t+H} + eps_next`
- Noise is scaled by dataset statistics:
  - latent noise uses `sqrt(latent_var)`
  - action noise uses `sqrt(action_var)`

## Loss and Energy

Training uses per-head reconstruction error with variance scaling:

- `state_energy = mean((z_hat - z_t)^2 / var_z)`
- `action_energy = mean((a_hat - a)^2 / var_a)`
- `next_state_energy = mean((z_next_hat - z_{t+H})^2 / var_z)`

The total DSM score is:

`u_t = state_energy + action_energy + next_state_energy`

Dataset variances are computed from expert trajectories and stored in the checkpoint as:

- `latent_mean`
- `latent_var`
- `action_mean`
- `action_var`

## Inference and Discriminator

File: `core/dsm_discriminator.py`

- Inference calls `model.forward(..., add_noise=False)`
- No Gaussian noise is added at test time
- Each transition produces three energy terms:
  - `state_energy`
  - `action_energy` / policy energy
  - `next_state_energy` / dynamics energy
- The step anomaly score is their sum:

`u_t = state_energy + policy_energy + dynamics_energy`

- Temporal smoothing is still handled by the existing `lambda_mode` and `lambda_window_size`

## Visualization Mapping

Files:

- `visualize_failures.py`
- `app/visualize.py`

The visualization pipeline now displays the DSM decomposition with:

- `state_error -> state_energy`
- `action_error -> policy_energy`
- `next_state_error -> dynamics_energy`

These values are passed through metadata and rendered in plots / videos using the `st`, `pi`, and `dy` labels.

## Notes

- The frozen policy / visual encoder remains unchanged
- KNN-based metrics are no longer used in the DSM detector path
- Attribution is now based on reconstruction energy decomposition rather than heuristic metric weights
