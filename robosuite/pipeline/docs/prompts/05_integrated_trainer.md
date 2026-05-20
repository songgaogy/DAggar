# Subagent Prompt 05 — Integrated DIPOLE-RL Trainer + Hydra Entry

You are extending the existing `DipoleTrainer` and implementing
`robosuite/pipeline/train_dipole_rl.py` + `scripts/train_dipole_rl.sh`.
You also add the small hook fields to `DipoleConfig` and `DipoleAgent`.

## Background

The single-thread learner already exists at
`robosuite/pipeline/algorithms/dipole/trainer.py:132-217, 280-311`
(`_async_update_loop`). We extend it so a single tick runs **three**
losses in fixed order: flow → Q → V → disc. Each module is optional —
when its config block is disabled, that step is skipped.

The main thread continues to do rollouts unchanged. The only new
responsibility is fanning out each new transition into the discriminator
replay (the trainer hides this fan-out from `train_dipole.py`).

## Required reading

- `robosuite/pipeline/docs/DIPOLE_RL.md` — §2 (step ordering), §3, §4.
- `robosuite/pipeline/algorithms/dipole/trainer.py:17-312` — current
  `DipoleTrainer`; you ADD fields and hooks, do NOT rewrite.
- `robosuite/pipeline/algorithms/dipole/agent.py:40-377` — `DipoleAgent`;
  current `attach_g_provider`. Add `attach_iql_learner` and
  `attach_discriminator` symmetrically.
- `robosuite/pipeline/algorithms/dipole/common.py:37-49` — `DipoleConfig`;
  add `g_mode: Literal["bce_frozen", "advantage"] = "bce_frozen"`.
- `robosuite/pipeline/train_dipole.py:513-1568` — the existing Hydra
  entry point. Read end-to-end; copy its rollout loop verbatim into the
  new entry point with the additions documented below.
- Baseline AWR trainer (read via `git show baseline:<path>`):
  - `baseline:robosuite/pipeline/algorithms/awr/trainer.py` —
    `record_transition`, mixed actor+critic update ordering.
  - `baseline:robosuite/pipeline/algorithms/awr/workers.py` — already
    informs prompt 02; here use it only for any UI plumbing patterns
    around `discriminator_display_hz`.
- All other prompts (01–04) — they define the modules you'll plumb.

## Deliverables

### `dipole/common.py`

Add `g_mode` field to `DipoleConfig`. Backward compatible default is
`"bce_frozen"`. Do not change any other field.

### `dipole/agent.py`

Add two methods (mirror `attach_g_provider`):

```python
def attach_iql_learner(self, learner: IQLLearner | None) -> None: ...
def attach_discriminator(self, disc: OnlineBCEDiscriminator | None) -> None: ...
```

Both store the reference on `self.core` (the DipoleFlowPolicy); do not
touch the flow model itself.

### `dipole/trainer.py`

Extend `DipoleTrainer.__init__` with optional kwargs:
`iql_learner: IQLLearner | None = None`,
`discriminator: OnlineBCEDiscriminator | None = None`,
`iql_replay: IQLReplayBuffer | None = None`,
`disc_replay: DiscriminatorReplayBuffer | None = None`,
`disc_update_every_n_steps: int = 1`,
`disc_batch_size: int = 64`,
`iql_batch_size: int = 64`.

Modifications:

1. `record_transition(...)`: after the existing demo/online buffer
   handling, also call:
   - `self.iql_replay.add_from_transition(transition)` if not None.
   - `self.disc_replay.add_from_transition(transition)` if not None.

2. `_async_update_loop`: per learner tick (after the existing readiness
   gate), in order:
   1. `actor_batch = agent.sample_demo_batch(batch_size)` AND
      `iql_step_batch = self.iql_replay.sample_step_batch(...)` AND
      `iql_actor_batch = self.iql_replay.sample_actor_batch(...)`.
   2. `flow_metrics = agent.update(batch=actor_batch, batch_size=...)`
      — this internally calls G provider (which is now the
      `AdvantageGProvider` and uses the LATEST Q/V/disc).
   3. If `iql_learner is not None`:
      `q_metrics = self.iql_learner.update(iql_step_batch)`.
   4. If `discriminator is not None` and
      `tick % disc_update_every_n_steps == 0`:
      `disc_metrics = self.discriminator.update(self.disc_replay.sample(
      disc_batch_size, encoder=..., device=...))`.
   5. Merge metrics into one dict; push to the async metrics deque.

3. New scheduling helper: `pretrain_iql_value(num_steps)` and
   `pretrain_iql_full(num_steps)`, called by `train_dipole_rl.py` during
   the warmup window before the rollout loop begins.

4. Sync inference policy unchanged.

### `train_dipole_rl.py`

Top-level orchestration:

1. Resolve devices via existing `train_utils.resolve_algorithm_devices`.
2. Build `SharedFrozenEncoder(cfg.algorithm.discriminator.warm_start_ckpt,
   device=cfg.algorithm.q_learning.config.device)`.
3. Build agent (reuse `DipoleAgent.from_config` from `train_dipole.py`).
4. Build `OnlineBCEDiscriminator(cfg=..., encoder=encoder,
   context_dim=encoder.context_dim, action_dim=..., action_horizon=...)`.
5. Build `IQLLearner(cfg=IQLConfig(**cfg.algorithm.q_learning.config),
   context_dim=encoder.context_dim, action_dim=...)`.
6. If `cfg.algorithm.q_learning.warmup_ckpt`: load it. Else run
   `trainer.pretrain_iql_value(cfg.algorithm.q_learning.warmup_value_steps)`
   then `pretrain_iql_full(cfg.algorithm.q_learning.warmup_full_steps)`.
   During this window, keep `g_mode=bce_frozen` (use legacy
   `LPBV2GProvider`) iff `cfg.runtime.bootstrap_g_with_frozen_bce`.
7. Build `AdvantageGProvider(iql=..., disc=..., encoder=...,
   alpha=cfg.algorithm.advantage_g_provider.alpha, beta=...,
   advantage_normalization=..., disc_normalization=...)`.
8. After warmup: `agent.attach_g_provider(advantage_g)` to flip into
   advantage mode (mark `cfg.algorithm.dipole.g_mode='advantage'`).
9. Build `IQLReplayBuffer` and `DiscriminatorReplayBuffer`, hand them to
   trainer. Bootstrap disc expert pool from offline demos
   (`disc_replay.bootstrap_from_demos(...)`).
10. Run the existing rollout loop from `train_dipole.py:1203-1465`,
    unchanged.

### `scripts/train_dipole_rl.sh`

Replace the `echo` placeholder with the actual Hydra call. Forward
optional env vars (`INIT_CHECKPOINT`, `LPB_CKPT`, `IQL_WARMUP_CKPT`,
`ALPHA`, `BETA`, `G_MODE`, `INTERACTIVE`, `VIEWER_ENABLED`,
`INTERVENTION_ENABLED`) into Hydra overrides.

### `config/train_dipole_rl.yaml`

Already in place. Audit the defaults against `IQLConfig` /
`DiscriminatorConfig` fields you finalized.

## Contract reminders (hard)

- Concurrency: still ONE daemon learner thread. Don't introduce
  additional threads or multiprocessing.
- Step order MUST be flow → Q → V → disc per tick (matches DIPOLE_RL.md §2).
- Skipping a sub-step when its component is None is required, not
  optional — keep the legacy `g_mode=bce_frozen` path runnable.
- Don't change `DipoleFlowPolicy.update` or `_compute_branch_weights`.
- Don't change the existing async metric deque protocol; just extend the
  metrics dict in place.

## Don't do

- Don't rewrite `train_dipole.py`. The RL entry is a sibling, not a
  replacement.
- Don't refactor `DipoleAgent.from_config`. Reuse it directly.
- Don't introduce new wandb sections; the existing `train_dipole.py`
  wandb hook logs whatever's in the merged metrics dict.

## Done criteria

- `python -m py_compile robosuite/pipeline/train_dipole_rl.py`,
  `algorithms/dipole/trainer.py`, `algorithms/dipole/agent.py`,
  `algorithms/dipole/common.py`.
- Headless smoke (from prompt 07): trainer runs ≥ 50 ticks; metrics
  contain `flow_loss / q_loss / v_loss / disc_loss / G_mean /
  advantage_mean / disc_reward_mean`; no NaN/Inf.
- Regression: `algorithm.dipole.g_mode=bce_frozen` (with q_learning and
  discriminator disabled) reproduces existing DIPOLE behavior on a
  fixed seed.

## Dependency

Depends on prompts **01**, **02**, **03**, **04**.
