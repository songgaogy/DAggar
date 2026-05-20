# Subagent Prompt 04 — AdvantageGProvider

You are implementing
`robosuite/pipeline/algorithms/dipole/advantage_g_provider.py`.
Skeleton is in place; fill in the `NotImplementedError` bodies.

## Background

The DIPOLE flow policy reads its branch-weighting signal G from
`g_provider.compute_g_for_batch(dipole_batch) -> (B,)`. Today the only
provider is the frozen `LPBV2GProvider`. DIPOLE-RL needs a new provider
that mixes:

- The IQL Q-chunking advantage `A(s, a) = min(Q1, Q2) - V`.
- The online BCE discriminator logit `disc_logit`.

into

```
G = α · z(A) + β · (−z(disc_logit))
```

where `z(·)` is the configured normalization. The minus sign on
`disc_logit` keeps DIPOLE's "higher G = more expert-like" convention
intact (the polarity sigmoid is unchanged).

## Required reading

- `robosuite/pipeline/docs/DIPOLE_RL.md` — §2 (G composition), §4 (API).
- `robosuite/pipeline/algorithms/dipole/g_provider.py:69-303` — the
  contract you must match (`compute_g_for_batch`, `compute_g_for_observation`,
  `bind_policy_cameras`).
- `robosuite/pipeline/algorithms/dipole/models/flow.py:343-394` —
  `_compute_branch_weights` consumes the G tensor; do not change anything
  there.
- `robosuite/pipeline/algorithms/dipole/common.py:62-88` — `DipoleBatch`
  schema; you read `image_obs_raw`, `proprio_raw`, `action_sequences_raw`.
- Prompts **01** and **02** for the IQL learner and discriminator APIs
  you call.

## Deliverables

- `__init__` stores the three injected dependencies (encoder, iql, disc)
  and the four scalar/string config knobs.
- `compute_g_for_batch(batch)` (no_grad):
  1. `context = self.encoder.encode(image_obs_raw=batch.image_obs_raw,
     proprio_raw=batch.proprio_raw)`.
  2. Build `IQLActorBatch(context=context,
     action_chunk_raw=batch.action_sequences_raw, metadata={})`.
  3. `advantage = self.iql_learner.compute_advantage_for_batch(actor_batch)`
     shape (B,).
  4. `disc_out = self.discriminator.score(context=context,
     action_chunk=batch.action_sequences_raw)`.
     `disc_logit = disc_out.logit` shape (B,).
  5. `A_norm = self._normalize(advantage, mode=self.advantage_normalization)`.
  6. `D_norm = self._normalize(disc_logit, mode=self.disc_normalization)`.
  7. `G = self.alpha * A_norm + self.beta * (-D_norm)`.
  8. Return `G` on the batch's device.
- `_normalize(x, mode)` implements `batch_zscore | running_zscore |
  minmax | none`. Mirror the four modes already living at
  `dipole/models/flow.py:343-394` so the conventions match exactly.
  Track running statistics in `self._running_advantage_*` and
  `self._running_disc_*` buffers (use the same EMA factor as DIPOLE,
  default 0.99).
- `compute_g_for_observation(...)`: optional; only implement if
  `eval_dipole.py` calls into it. If skipped, raise NotImplementedError
  with a clear message pointing the caller to the eval mode setup.
- `bind_policy_cameras(policy_cameras)` is a pass-through to the
  encoder.

## Contract reminders (hard)

- Output shape MUST be `(B,)`. Do not return `(B, 1)`.
- No grads through this object (entire call is `@torch.no_grad()`).
- Do not touch `DipoleFlowPolicy.update()` or its branch-weight logic.
- This object owns NOTHING that requires_grad. Q1/Q2/V/head parameters
  belong to their respective learners.

## Don't do

- Don't add new normalization modes beyond the four listed.
- Don't cache encoded contexts across calls. The encoder is called per
  batch; `compute_g_for_batch` is invoked in the flow-policy training
  loop where caching would risk staleness across optimizer steps.
- Don't ingest the discriminator's `decision` flag here; that's only
  for UI / human intervention. The mixing uses the raw logit.

## Done criteria

- `python -m py_compile` passes.
- Smoke test (write under `robosuite/pipeline/algorithms/dipole/tests/
  test_advantage_g_provider.py`): with fake IQL + fake disc returning
  fixed tensors, confirm G has expected sign/magnitude relationship to
  α and β.
- When this provider is attached, `DipoleFlowPolicy.update(batch)` runs
  without modification and the metrics dict still includes the standard
  DIPOLE fields plus the new metadata from the provider (the trainer
  prompt will log these separately — you don't have to expose them).

## Dependency

Depends on prompts **01_q_learning_iql**, **02_online_discriminator**,
**03_shared_encoder**.
