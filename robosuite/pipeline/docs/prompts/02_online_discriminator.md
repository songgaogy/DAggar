# Subagent Prompt 02 — Online BCE Discriminator

You are implementing all files under
`robosuite/pipeline/algorithms/discriminator/` except `encoder.py`
(handled by prompt 03) and `adapter.py` (already a placeholder). The
skeleton is in place.

## Background

Today the DIPOLE pipeline uses a **frozen** BCE discriminator
(`LPBV2GProvider`). DIPOLE-RL needs an **online-trainable** version that:

- Reuses the shared `SharedFrozenEncoder` (no encoder ownership here).
- Trains only the BCE head (`TrainableBCEHead`).
- Pulls positives from human-intervention transitions (`is_intervention=True`)
  and negatives from non-intervention online-policy rollouts.
- Provides `intrinsic_reward(context, action_chunk) -> (B,)` for Q
  learning to consume as `r_disc`.
- Emits a decision (above/below EMA threshold) for the UI / intervention
  display.

It is NOT conditioned on policy embeddings (user decision); the head
inputs are simply `concat(context, flatten(action_chunk))`.

## Required reading

- `robosuite/pipeline/docs/DIPOLE_RL.md` — §2, §4.
- `robosuite/pipeline/algorithms/discriminator/{base,bce_head,online_bce,
  replay,losses}.py` — skeletons.
- Current frozen reference:
  - `robosuite/pipeline/algorithms/dipole/g_provider.py:69-303` for the
    overall G provider shape (we mirror `score`/`intrinsic_reward` semantics
    but make the head trainable).
- Upstream BCE artifacts (READ ONLY — do not modify):
  - `robosuite/discriminator/lpb_v2/detectors/bce.py:76-107` —
    `BCEHead` architecture (LayerNorm + GELU MLP).
  - `robosuite/discriminator/lpb_v2/detectors/bce.py:380-455` — current
    score path, threshold lookup.
- Baseline online-disc reference (for UI/decision metadata only):
  `baseline:robosuite/pipeline/algorithms/awr/workers.py` —
  `AWRDiscriminatorLabel` carries `score / threshold / prediction /
  normalized_margin / metadata` fields. Mirror those in your
  `DiscriminatorOutput.metadata`.
- Transition definition: `robosuite/pipeline/common/types.py:14-37`.

## Deliverables

### `bce_head.py`

- Implement `TrainableBCEHead.forward` mirroring `BCEHead` from
  `lpb_v2/bce.py:76-107`: `[Linear(in, hidden) → LayerNorm → GELU] *
  num_layers → Linear(hidden, 1)`.
- Implement `warm_start_from_lpb_bce_ckpt`: load `bce_head.pth`, extract
  `bce_detector["head"]` (matches `BCEDiscriminator.state_dict()` schema
  in `bce.py:408-427`), `load_state_dict(strict=False)` so dim mismatches
  raise informatively. **Discard** thresholds and calibration — we
  re-estimate online.
- Init: Kaiming on hidden, zero on final.

### `online_bce.py`

- `__init__` stores the encoder reference, builds
  `TrainableBCEHead(in_dim=context_dim + action_dim*action_horizon, ...)`,
  builds AdamW over the head, initializes `self.threshold = 0.0`. If
  `cfg.warm_start_ckpt` is set, call `head.warm_start_from_lpb_bce_ckpt`.
- `score(context, action_chunk)` (no_grad):
  1. `head_in = concat(context, action_chunk.flatten(start_dim=1))`.
  2. `logit = head(head_in)`.
  3. `prob_expert = sigmoid(logit)`.
  4. `decision = logit > self.threshold` (boolean tensor).
  5. Pack into `DiscriminatorOutput` with `metadata = {"threshold":
     self.threshold, "normalized_margin": (logit - threshold).abs()}`.
- `intrinsic_reward(context, action_chunk)` (no_grad): returns the logit
  directly (higher = expert). The IQL config decides via
  `disc_reward_sign` whether to negate before adding to env reward.
- `update(batch: DiscriminatorBatch)`:
  1. `head_in` as above.
  2. `logit = head(head_in)` (grad on).
  3. `loss = bce_with_logits_loss(logit, batch.label, label_smoothing=
     cfg.label_smoothing)`.
  4. Backward, clip grad (use a sensible default like 1.0; expose if
     needed), step optimizer.
  5. Update EMA threshold using the policy-half of the batch's logits
     (median or 50th percentile; document choice).
  6. Return metrics: `disc_loss, disc_acc, expert_logit_mean,
     policy_logit_mean, threshold`.
- `state_dict / load_state_dict` round-trip head weights, optimizer
  state, threshold, and cfg.

### `replay.py`

- Two ring-buffer pools (`_Pool`) for expert (intervention) and policy
  (non-intervention) transitions.
- `add_from_transition(t)` routes by `t.is_intervention`. For offline
  bootstrap: `bootstrap_from_demos(demos)` adds demos into expert_pool
  with a low-probability mix (default: 25% of expert sampling is from
  the demo subset; you may implement as a tagged sub-pool to keep
  online recency).
- `sample(batch_size, encoder, device)`:
  1. `n_expert = round(batch_size * balance_ratio / (1 + balance_ratio))`.
  2. `n_policy = batch_size - n_expert`.
  3. If `len(expert_pool) < n_expert`, sample with replacement.
  4. Run frozen encoder on each side; concat into `DiscriminatorBatch`
     with `label = [1] * n_expert + [0] * n_policy`.
- `ready(batch_size)`: at least 1 expert and `n_policy` policy samples.

### `losses.py`

- `bce_with_logits_loss(logits, labels, label_smoothing, pos_weight)`:
  apply smoothing as `y' = y * (1 - 2s) + s` (so labels in [s, 1-s]),
  then `F.binary_cross_entropy_with_logits(logits, y', pos_weight=...)`.

### `base.py`

- No new work — confirm `DiscriminatorBase`, `DiscriminatorOutput`,
  `DiscriminatorBatch` match the contracts in DIPOLE_RL.md §4.

## Contract reminders (hard)

- The encoder is INJECTED (`__init__(..., encoder=...)`). Never call
  `encoder.parameters()` or set `requires_grad=True` on any encoder
  param.
- All inference paths under `torch.no_grad()`.
- `intrinsic_reward` returns shape `(B,)`, not `(B, 1)`.
- Output sign: `logit` is higher for more expert-like inputs. Q learner
  is responsible for the negation if needed (via `disc_reward_sign`).

## Don't do

- Don't condition the head on policy embeddings, snapshot ids, or
  rollout indices. User chose plain `(s, a)` BCE.
- Don't write the full BCE training loop with epochs/scheduler. This is
  online; each `update(batch)` is one optimizer step.
- Don't mutate `lpb_v2/bce.py`.

## Done criteria

- `python -m py_compile` over all `discriminator/*.py`.
- Unit test in `robosuite/pipeline/algorithms/discriminator/tests/
  test_online_bce.py` covering:
  - `score` shape / device.
  - `update` reduces BCE loss on a fixed synthetic 64-sample batch.
  - `warm_start_from_lpb_bce_ckpt` loads at least one Linear weight (you
    can read the pre-fitted ckpt at
    `checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth`).
  - `DiscriminatorReplayBuffer.sample` returns the requested balance.

## Dependency

Depends on prompt **03_shared_encoder** (uses `SharedFrozenEncoder`).
Parallel-safe with **01_q_learning_iql**.
