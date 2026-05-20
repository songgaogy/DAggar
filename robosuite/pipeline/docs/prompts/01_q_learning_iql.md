# Subagent Prompt 01 — IQL Q-chunking Learner

You are implementing all files under
`robosuite/pipeline/algorithms/q_learning/` (skeleton in place).
Mirror the baseline AWR implementation but adapt to DIPOLE-RL's API.

## Background

DIPOLE-RL needs an IQL learner with **Q-chunking** (Q evaluates an entire
H-step action chunk holistically, not per-step). The actor is the DIPOLE
flow policy and is NOT updated by this module — Q/V exclusively produce
the advantage signal that `AdvantageGProvider` mixes into G. There is no
SAC actor step and no log_pi computation, because flow policies don't
expose a tractable density.

The reward going into Bellman targets is `r_total = r_env + λ_disc · r_disc`,
where `r_disc` is re-evaluated at sample time against the **current**
discriminator (no stale rewards).

## Required reading

- `robosuite/pipeline/docs/DIPOLE_RL.md` — §2 (per-tick step order),
  §3 (module layout), §4 (API contract). Note the prohibitions in §4.
- `robosuite/pipeline/algorithms/q_learning/{common,networks,losses,
  replay,iql,warmup,data_util}.py` — skeletons. Don't change signatures.
- Baseline AWR reference (read via `git show baseline:<path>` — these
  files don't exist on the current branch):
  - `baseline:robosuite/pipeline/algorithms/awr/common.py` — pattern for
    `IQLConfig / IQLStepBatch / IQLActorBatch`.
  - `baseline:robosuite/pipeline/algorithms/awr/models/flow.py:73-111`
    — Q1, Q2, V build (`_build_mlp`).
  - `baseline:robosuite/pipeline/algorithms/awr/models/flow.py:37-39`
    — expectile loss.
  - `baseline:robosuite/pipeline/algorithms/awr/models/flow.py:266-295`
    — `update_value` (Bellman + expectile).
  - `baseline:robosuite/pipeline/algorithms/awr/replay_buffer.py:295-510`
    — chunk-window sampling, n-step reward aggregation.
- Existing DIPOLE replay you'll wrap:
  `robosuite/pipeline/algorithms/dipole/replay_buffer.py:22-143`.
- Transition data class (already has reward/next_obs/done):
  `robosuite/pipeline/common/types.py:14-37`.

## Deliverables

### `common.py`

- Confirm field order / names match the skeleton.
- Implement `IQLStepBatch.to(device)` and `IQLActorBatch.to(device)`:
  move all tensors; deep-copy metadata is unnecessary.

### `networks.py`

- `QChunkNetwork`: `_build_mlp` over `(context_dim + action_dim *
  action_horizon, hidden_dims..., 1)` with LayerNorm + GELU between
  hidden layers (mirror baseline awr/models/flow.py:73-111).
- `VNetwork`: same but input is `context_dim` only.
- Both apply Kaiming init on hidden layers and zero-init the final layer
  (helps Q/V calibration).

### `losses.py`

- `bellman_q_loss(q_pred, target_q) = F.mse_loss(q_pred, target_q)`.
- `expectile_v_loss(diff, tau)` — `weight = where(diff > 0, tau, 1-tau)`;
  loss = mean(weight * diff^2).
- `compute_advantage(q1, q2, v)` — return `(min(q1, q2) - v).squeeze(-1)`.

### `replay.py`

- `IQLReplayBuffer.__init__` stores a reference to the underlying
  transition store (don't copy frames).
- `sample_step_batch`:
  1. Sample `batch_size` valid contiguous chunks of length
     `cfg.action_horizon` (all H steps in the same episode; reject windows
     that cross episode boundaries).
  2. Build raw image / proprio tensors for `s` and `s'`
     (s' = transition at chunk start + H).
  3. `with torch.no_grad():` run `encoder.encode(...)` for both to get
     `context` and `next_context`.
  4. Per-step env rewards → `r_env_chunk` (B, H).
  5. `with torch.no_grad():` compute
     `r_disc_step = discriminator.intrinsic_reward(context, action_chunk)`
     ONCE per sample (treat the chunk as the unit; do not call disc
     H times). Broadcast to (B, H) by dividing by H (or just put it on
     the first step — your call; document in the implementation).
  6. `r_total_chunk = r_env_chunk + cfg.disc_reward_coef * r_disc_chunk`.
  7. `rewards = data_util.aggregate_chunk_reward(r_total_chunk, cfg.discount)`.
  8. `dones = data_util.chunk_done_mask(step_dones)`.
  9. Pack into `IQLStepBatch` on `device`.
- `sample_actor_batch`: lighter version; no s', no reward, no done. Used
  by `AdvantageGProvider`.
- `ready(batch_size)`: enough valid chunk start indices.

### `iql.py`

- `__init__` instantiates `q1, q2, v, target_v` (target_v = clone of v;
  set `requires_grad=False` on target params).
- Two optimizers:
  - `q_optim = AdamW(q1.params + q2.params, lr=cfg.q_lr, wd=cfg.weight_decay)`
  - `v_optim = AdamW(v.params, lr=cfg.v_lr, wd=cfg.weight_decay)`
- `update(step_batch)`:
  1. `with torch.no_grad():` `target_q = step_batch.rewards +
     (cfg.discount ** cfg.action_horizon) * (1 - step_batch.dones) *
     self.target_v(step_batch.next_context)`.
  2. `q1_pred = self.q1(context, action_chunk); q2_pred = self.q2(...)`.
  3. `q_loss = bellman_q_loss(q1_pred, target_q) + bellman_q_loss(q2_pred,
     target_q)`. Backward, clip to `cfg.grad_clip_norm`, step.
  4. `with torch.no_grad():` `q_min = min(q1_pred.detach(), q2_pred.detach())`.
  5. `v_pred = self.v(context)`; `v_loss = expectile_v_loss(q_min - v_pred,
     cfg.expectile_tau)`. Backward, clip, step.
  6. Polyak: `target_v ← (1-τ_p)·target_v + τ_p·v`.
  7. Return metrics dict (loss values, q/v means, td_error mean).
- `warmup_value_only(step_batch)`: same as `update` but **skip** the Q
  update; V regresses toward `r_total + γ^H · (1-done) · target_v(next)`
  directly (no min-of-two-Q). Used during pre-Q warmup.
- `compute_advantage_for_batch(actor_batch)` (no_grad): returns
  `compute_advantage(q1, q2, v)` shape (B,).
- `state_dict / load_state_dict` round-trip Q1/Q2/V/target_V + optimizers
  + cfg.

### `warmup.py`

- Argparse: `--config`, `--output`, `--value-steps`, `--full-steps`,
  `--device`.
- Build encoder, disc (warm-start from `algorithm.discriminator.
  warm_start_ckpt`), `IQLLearner`. Load offline demos + success/failure
  rollout caches into the shared transition store.
- Run `value_steps` of `warmup_value_only` then `full_steps` of `update`.
- Save state to `<output>/iql_state.pt`.

### `data_util.py`

- `aggregate_chunk_reward`: explicit Python loop or `torch.pow` × cumprod;
  test on H=1 and H=8.
- `chunk_done_mask`: `any(dim=1, keepdim=True)`.
- `tile_proprio_as_action`: repeat proprio to fill `action_input_dim`;
  pad with zeros if proprio_dim doesn't divide cleanly (document this).

## Contract reminders (hard)

- Q input shape is `(B, D_ctx + H·D_a)`. No per-step Q.
- The encoder is INJECTED; never instantiate it here.
- Every encoder/disc forward is under `torch.no_grad()`.
- Both samplers must support `replay.ready(batch_size)` so the trainer
  can gate updates safely.
- Per the DIPOLE-RL prohibitions: no `torch.multiprocessing`, no DDP, no
  changes to `DipoleFlowPolicy.update` signature.

## Done criteria

- `python -m py_compile` over all `q_learning/*.py`.
- Standalone smoke test: load `<repo>/data/demo_handful.h5` (or any
  small offline cache present in the repo), run `warmup.py --value-steps
  20 --full-steps 20` on a CPU-only fake encoder mock (write a tiny test
  encoder if needed); confirm no NaN.
- Unit test in `robosuite/pipeline/algorithms/q_learning/tests/test_iql.py`
  covering `compute_advantage`, `expectile_v_loss`, and a single
  `iql.update` step with synthetic tensors.

## Dependency

Depends on prompt **03_shared_encoder** (uses `SharedFrozenEncoder` via
DI). May run in parallel with **02_online_discriminator** since you
import disc only via `TYPE_CHECKING`.
