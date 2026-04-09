# LPB Score Conditional Fisher DSM

This module implements an offline failure detector over latent transitions from the frozen flow policy encoder.

## What Changed

The refactored `lpb_score` no longer uses heteroscedastic NLL reconstruction. It now:

- trains a unified conditional DSM with a binary trajectory-type condition
- uses `traj_type = 0` for positive data from `expert + success_rollout`
- uses `traj_type = 1` for negative data from `fail_rollout`
- scores each step with Fisher-style L2 distances between the positive and negative conditional predictions

Only `train + visualize` remain in this module. The old standalone `analyse` and `eval` entrypoints were removed.

## Data Layout

Each task still uses the same on-disk folders:

- `./data/<task_name>/expert`
- `./data/<task_name>/success_rollout`
- `./data/<task_name>/fail_rollout`

Training positives are sampled from the merged `expert + success_rollout` pool for each task. Negatives are sampled from `fail_rollout`.

The split config now uses:

- `num_pos_traj`
- `num_neg_traj`

instead of separate expert / success / fail counts.

## Model

File: `core/model.py`

The unified transformer still routes three denoising factors through one shared backbone:

- state branch
- action branch
- dynamics branch

Task routing still uses a learned task token. The refactor adds:

- `type_embedding = nn.Embedding(2, embed_dim)`

The route token becomes:

- `task_embedding(task_id) + type_embedding(traj_type)`

The model output head is now deterministic:

- `pred_mean`

There is no `pred_logvar`.

## Training Objective

Training still happens in normalized latent / action space with Gaussian noise injection.

For each sampled transition:

1. normalize `z_t`, `a_{t:t+H-1}`, and `z_{t+H}`
2. build the routed clean targets
3. add Gaussian noise to the routed targets
4. predict the clean routed targets conditioned on `traj_type`
5. optimize plain MSE

The batch loss is the mean of:

- state MSE
- action MSE
- dynamics MSE

Normalization statistics are computed from the sampled positive training refs, not expert-only refs.

## Inference Score

File: `core/dsm_discriminator.py`

At inference time the detector runs the same transition twice:

1. once with `traj_type = 0`
2. once with `traj_type = 1`

For each branch, the detector computes a Fisher-style squared L2 distance:

- `state_error = ||state_pos - state_neg||^2`
- `action_error = ||action_pos - action_neg||^2`
- `next_state_error = ||dynamics_pos - dynamics_neg||^2`

The final per-step score is:

- `u_t = state_error + action_error + next_state_error`

The existing temporal aggregation logic remains unchanged:

- `lambda_mode = mean` or `max`
- `lambda_window_size` controls prefix or rolling aggregation

Thresholds are still calibrated inside `visualize` from positive bank trajectories.

## Entrypoints

### Train

Python entry:

- `robosuite/discriminator/lpb_score/train.py`

Bash entry:

- `robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm.sh`

Important environment overrides:

- `NUM_POS`
- `NUM_NEG`
- `POSITIVE_RATIO`
- `CKPT`
- `GPU`

Example:

```bash
NUM_POS=240 NUM_NEG=120 GPU=0 \
bash robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm.sh
```

### Visualize

Python entry:

- `robosuite/discriminator/lpb_score/visualize_failures.py`

Bash entry:

- `robosuite/discriminator/lpb_score/scripts/visualize_lpb_score_dsm_failures.sh`

The script accepts `DSM_CKPT` explicitly, otherwise it auto-discovers the latest final checkpoint under `checkpoints/multitask_6/lpb_score`.

## Notes

- `state_error_scores`, `action_error_scores`, and `next_state_error_scores` are kept for compatibility in summaries and visualization code, but they now mean Fisher L2 energies.
- Old `lpb_score` checkpoints are intentionally incompatible with this refactor because the variance head was removed and `type_embedding` was added.
