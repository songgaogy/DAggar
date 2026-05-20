# Subagent Prompt 06 — Offline Q/V Warmup CLI

You are filling in
`robosuite/pipeline/algorithms/q_learning/warmup.py`.
The skeleton has a `main()` stub.

## Background

Online RL is unstable when V is uncalibrated and the discriminator's
intrinsic reward is noisy. The warmup script bootstraps Q/V from offline
demos + cached success/failure rollouts before online training begins.

It produces `iql_state.pt`, which `train_dipole_rl.py` can load via
`algorithm.q_learning.warmup_ckpt` to skip warmup on subsequent runs.

## Required reading

- `robosuite/pipeline/docs/DIPOLE_RL.md` — §2 (per-tick step order;
  warmup uses `warmup_value_only` then `update`).
- `robosuite/pipeline/algorithms/q_learning/iql.py` and `warmup.py`
  skeletons.
- Baseline reference: `baseline:robosuite/pipeline/algorithms/awr/models/
  build_awr_qv_cache.py` — exact warmup pattern; copy structure.
- DIPOLE offline demo loader: `robosuite/pipeline/train_dipole.py:702-909`
  (the helper functions there will be reusable).

## Deliverables

`main()` implementation:

1. Parse args via argparse (no Hydra here — keep the CLI lean):
   - `--config` (Hydra config yaml; we read the same train_dipole_rl.yaml).
   - `--output` (default `outputs/iql_warmup/iql_state.pt`).
   - `--value-steps` (default from
     `algorithm.q_learning.warmup_value_steps`).
   - `--full-steps` (default from `algorithm.q_learning.warmup_full_steps`).
   - `--device` (default cuda:1).
   - `--demo-h5` (optional override of demo HDF5 path).
   - `--success-cache`, `--fail-cache` (optional rollout caches).

2. Load yaml + apply CLI overrides.

3. Build `SharedFrozenEncoder` from
   `algorithm.discriminator.warm_start_ckpt`.

4. Build `OnlineBCEDiscriminator` with `online_train=False` for the
   duration of warmup (treat it as frozen so the intrinsic reward is at
   least consistent during the warmup window). Warm-start from
   `bce_head.pth`.

5. Build the underlying transition store and bootstrap from
   `--demo-h5` (+ success/fail caches). Reuse `train_dipole.py`'s demo
   loader helpers; do NOT re-implement HDF5 parsing.

6. Build `IQLReplayBuffer` over the transition store.

7. Build `IQLLearner(IQLConfig(**cfg.algorithm.q_learning.config),
   context_dim=encoder.context_dim, action_dim=...)`.

8. Warmup loop:
   ```python
   for step in range(value_steps):
       batch = iql_replay.sample_step_batch(batch_size, encoder=encoder,
                                            discriminator=disc, device=device)
       metrics = iql.warmup_value_only(batch)
       if step % 100 == 0:
           log(metrics)
   for step in range(full_steps):
       batch = iql_replay.sample_step_batch(...)
       metrics = iql.update(batch)
       if step % 100 == 0:
           log(metrics)
   ```

9. Save `iql.state_dict()` plus encoder metadata
   (`{"context_dim": encoder.context_dim, "action_dim": cfg.action_dim,
   "iql_cfg": asdict(cfg)}`) to `<output>` via `torch.save`.

## Contract reminders

- Encoder & disc are frozen here (`torch.no_grad` on all forwards).
- Reward composition matches the online path: `r_total = r_env +
  λ_disc · r_disc`. Same code path, just a frozen discriminator.

## Don't do

- Don't run env rollouts. Warmup is offline only.
- Don't write Hydra overrides to disk; output is only the .pt file.
- Don't run for more than `--value-steps + --full-steps` iterations
  (no early-stopping heuristics).

## Done criteria

- `python -m robosuite.pipeline.algorithms.q_learning.warmup
  --value-steps 200 --full-steps 200 --demo-h5 <small_demo>.h5
  --device cuda:1` finishes without errors, produces a non-empty
  `iql_state.pt` (size > 1 MB).
- V loss roughly decreases over the value warmup window (log to stdout).

## Dependency

Depends on prompts **01**, **02**, **03**, **05** (so the trainer-side
helpers and replay buffers exist).
