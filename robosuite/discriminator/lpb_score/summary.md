# LPB Score Refactor Summary

This document summarizes the current `robosuite/discriminator/lpb_score` design after the conditional Fisher DSM refactor.

## Goal

The detector operates on latent transitions:

- current latent `z_t`
- action chunk `a_{t:t+H-1}`
- future latent `z_{t+H}`

It estimates whether a transition is more consistent with the positive manifold or the failure manifold.

## Data Semantics

Per task, the module uses three raw data families:

- `expert`
- `success_rollout`
- `fail_rollout`

For split construction:

- positives = pooled `expert + success_rollout`
- negatives = `fail_rollout`

The split config uses:

- `num_pos_traj`
- `num_neg_traj`

Sampling is done independently per task. Positive refs are shuffled once from the merged pool and then sliced into non-overlapping `train / val / test` subsets. Negative refs are handled the same way from `fail_rollout`.

The transition dataset emits:

- `current_latent`
- `action_sequence`
- `target_latent`
- `traj_type`

with:

- `traj_type = 0` for positive
- `traj_type = 1` for negative

## Model

The model is still a unified transformer over three routed denoising tasks:

1. state
2. action
3. dynamics

The refactor adds trajectory-type conditioning:

- `task_embedding(task_id)`
- `type_embedding(traj_type)`

These are summed before entering the shared backbone.

The output head is deterministic:

- only `pred_mean`

There is no variance head and no `logvar` optimization.

## Training

Training remains denoising-style in normalized space:

1. standardize latent and action features
2. build clean routed targets
3. add Gaussian noise to the routed targets
4. predict clean routed targets conditioned on `traj_type`
5. optimize plain MSE

Normalization stats are computed from the sampled positive training refs.

Weighted batch sampling still exists, but it now balances:

- positive samples
- negative samples

through `training.positive_ratio`.

## Inference

Inference is no longer reconstruction scoring. For each transition, the model runs:

1. positive-conditioned forward pass with `traj_type = 0`
2. negative-conditioned forward pass with `traj_type = 1`

Per-branch Fisher energies are:

- `state_error = ||state_pos - state_neg||^2`
- `action_error = ||action_pos - action_neg||^2`
- `next_state_error = ||dynamics_pos - dynamics_neg||^2`

The step score is:

- `u_t = state_error + action_error + next_state_error`

The existing `lambda_t` aggregation and threshold calibration logic are unchanged.

## Runtime Surface

The module now supports two workflows:

- training
- visualization

Removed from the module:

- standalone analyse entrypoint
- standalone eval entrypoint
- their YAML configs
- their bash scripts

Visualization still performs self-calibration from the configured positive bank trajectories before rendering fail trajectories.

## Practical Notes

- Compatibility keys such as `state_error_scores` are still present in detector metadata and summaries, but they now store Fisher L2 energies.
- Old checkpoints from the previous heteroscedastic DSM are intentionally incompatible with the new architecture.
- The training bash script exposes `NUM_POS` and `NUM_NEG` so per-task training counts can be overridden without editing YAML.
