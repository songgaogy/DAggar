# DIPOLE-RL Current Implementation Guide

This document describes the current DIPOLE-RL implementation under `robosuite/pipeline`. It is written for coding agents that need to modify, debug, or extend the code without reading the older subagent logs.

## 1. Current System

DIPOLE-RL extends DIPOLE with:
- IQL-style Q/V learning over action chunks.
- A Q ensemble with `K=5` critics.
- A single-frame online BCE discriminator aligned with LPB v2.
- A shared frozen LPB encoder injected into Q-learning, discriminator, and AdvantageG.
- Optional offline Q/V warmup with preencoded replay cache.

Main entry points:
- Training: `robosuite/pipeline/train_dipole_rl.py`
- Config: `robosuite/pipeline/config/train_dipole_rl.yaml`
- Training wrapper: `robosuite/pipeline/scripts/train_dipole_rl.sh`
- Offline Q/V warmup: `robosuite/pipeline/algorithms/q_learning/warmup.py`
- Warmup wrapper: `robosuite/pipeline/scripts/utils/init_iql_qv.sh`

Core modules:
- `algorithms/discriminator/encoder.py`: `SharedFrozenEncoder`
- `algorithms/q_learning/iql.py`: `IQLLearner`
- `algorithms/q_learning/replay.py`: `IQLReplayBuffer`, `IQLPreencodedReplayCache`
- `algorithms/discriminator/online_bce.py`: `OnlineBCEDiscriminator`
- `algorithms/discriminator/replay.py`: `DiscriminatorReplayBuffer`
- `algorithms/dipole/advantage_g_provider.py`: `AdvantageGProvider`
- `algorithms/dipole/trainer.py`: fused online learner step

## 2. Data Flow

Online training tick:
1. The main rollout loop records transitions through `DipoleTrainer.record_transition`.
2. `IQLReplayBuffer.sample_step_batch` samples H-step chunks from the online buffer.
3. `SharedFrozenEncoder.encode_chunk_frames` encodes all chunk frames with real actions.
4. IQL updates all Q critics and V.
5. `DiscriminatorReplayBuffer.sample` samples single frames and updates the BCE head.
6. `AdvantageGProvider.compute_g_for_batch` computes `G` for the DIPOLE flow update.
7. `DipoleFlowPolicy.update()` remains unchanged; it consumes `G` only through the provider interface.

Reward for IQL:
```text
r_total[t] = output_reward_coef * r_env[t] + disc_reward_coef * r_disc[t]
r_disc[t] = -sigmoid(failure_score[t] - tau)
```

`r_disc` is in `(-1, 0)`: failure-like frames are near `-1`, expert-like frames are near `0`. `aggregate_chunk_reward` gamma-aggregates the per-step rewards over the action horizon.

Offline warmup:
- Loads HDF5 demos into a `FlowDaggerReplayBuffer`.
- Annotates transitions with LPB v2 benchmark discriminator rewards when `disc_reward_coef != 0`.
- Builds an optional `IQLPreencodedReplayCache` so warmup loops do not repeatedly stack images and re-run encoder forwards.
- Saves `iql_state.pt` for online training via `algorithm.q_learning.warmup_ckpt`.

## 3. Q Learning

Config fields live in `IQLConfig`:
```yaml
algorithm:
  q_learning:
    config:
      action_horizon: ${algorithm.flow.action_horizon}
      discount: 0.99
      expectile_tau: 0.7
      q_ensemble_size: 5
      v_subset_size: 2
      q_lr: 3.0e-4
      v_lr: 3.0e-4
      target_polyak: 0.005
      hidden_dims: [512, 512]
      output_reward_coef: 0
      disc_reward_coef: 1.0
      update_freq: 1
```

Current Q/V behavior:
- Q input: `concat(context, flatten(action_chunk))`, shape `D_ctx + H * D_a`.
- V input: `context`, shape `D_ctx`.
- Q ensemble size: `K=5`.
- Q update: train all 5 critics against the same Bellman target; loss is averaged over the ensemble.
- V update: randomly select `M=2` critics each update, compute subset min, then expectile-regress V.
- Advantage for G: `A(s, a) = mean(Q_1..Q_5) - V`.
- Target network: `target_v` only; there is no target Q network.

Checkpoint behavior:
- New Q checkpoints use the `q_ensemble` schema.
- Old `{q1, q2}` checkpoints are intentionally rejected. Re-run offline warmup after enabling Q ensemble.
- `q1` and `q2` still exist as read-only aliases for visualization/tests, but persistence is ensemble-only.

## 4. Discriminator

`OnlineBCEDiscriminator` is single-frame and LPB v2 aligned:
- Input to the head: `context: (B, D_ctx)`.
- The real action enters through `SharedFrozenEncoder.encode(..., action_real=...)`.
- The BCE head no longer consumes flattened action chunks.
- Warm-start from LPB v2 should load the first Linear layer; shape mismatch is treated as a regression.

Sign convention:
```text
expert_logit = head(context)
failure_score = -expert_logit
tau = bce_youden_threshold
prob_failure = sigmoid(failure_score - tau)
r_disc = -prob_failure
```

Training labels:
- `label=1`: intervention / failure frame.
- `label=0`: expert demo or non-intervention frame.
- Online BCE maps these to LPB expert targets with `expert_target = 1 - label`.

API:
```python
disc.score(context=ctx).logit          # expert_logit, shape (B,)
disc.score(context=ctx).prob_failure   # failure probability, shape (B,)
disc.intrinsic_reward(context=ctx)     # r_disc in (-1, 0), shape (B,)
disc.update(discriminator_batch)       # one BCE head update
```

Do not pass `action_chunk` to `score` or `intrinsic_reward`; that API is obsolete.

## 5. Shared Encoder Rules

`SharedFrozenEncoder` is owned by the trainer and injected everywhere else. Do not instantiate private encoder copies inside IQL, discriminator, or AdvantageG.

Hard rules:
- Call `bind_policy_cameras(...)` once before any encode.
- Keep encoder frozen: no `train()`, no gradients.
- Use `torch.no_grad()` for all encoder forwards.
- Do not pre-crop or pre-resize images before calling the shared encoder; it owns LPB v2 preprocessing.
- For LPB-aligned discriminator/Q replay contexts, pass real actions with `action_real`.
- The degraded `action_real=None` path exists only for legacy display/fallback paths.

Important encoder methods:
```python
encoder.encode(
    image_obs_raw=(B,V,3,H,W),
    proprio_raw=(B,D_s),
    action_real=(B,D_a) | None,
) -> (B,D_ctx)

encoder.encode_chunk_frames(
    chunk_images=(B,H,V,3,H_img,W_img),
    chunk_proprio=(B,H,D_s),
    chunk_actions=(B,H,D_a),
) -> (B,H,D_ctx)
```

## 6. Advantage G

`AdvantageGProvider` computes:
```text
A = mean(Q_1..Q_5) - V
disc_logit = disc.score(context).logit
G = alpha * norm(A) + beta * norm(-disc_logit)
```

The sign on `-disc_logit` is intentional. `disc.score(...).logit` is the LPB expert logit; negating it makes higher values more failure-like, matching DIPOLE polarity weighting.

The provider implements only batch scoring:
- `compute_g_for_batch(batch) -> (B,)`
- `compute_g_for_observation(...)` raises `NotImplementedError`

Keep the legacy `LPBV2GProvider` for:
- `g_mode=bce_frozen`
- live discriminator/display overlays
- bootstrap before switching to advantage mode

## 7. Config Summary

`train_dipole_rl.yaml` inherits `train_dipole.yaml`.

Key fields:
```yaml
algorithm:
  dipole:
    g_mode: "advantage"        # "advantage" | "bce_frozen"
  q_learning:
    enabled: true
    warmup_value_steps: 20000
    warmup_full_steps: 10000
    warmup_ckpt: null
  discriminator:
    enabled: true
    online_train: true
    warm_start_ckpt: "checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth"
  advantage_g_provider:
    alpha: 1.0
    beta: 0.5
    advantage_normalization: "batch_zscore"
    disc_normalization: "batch_zscore"

warmup:
  demo_splits: [expert, success_rollout, fail_rollout]
  preencode_cache: true
  preencode_batch_size: 64
  preencode_cache_device: "cpu"   # use "learner" only if GPU memory allows

runtime:
  init_checkpoint: null
  bootstrap_g_with_frozen_bce: true
```

Notes:
- `algorithm.q_learning.enabled=false` disables IQL construction and warmup.
- `algorithm.discriminator.online_train=false` disables online BCE construction/update.
- `g_mode=advantage` requires both IQL and online discriminator enabled.
- `algorithm.dipole.lpb_detector.ckpt_path` should stay aligned with `algorithm.discriminator.warm_start_ckpt`.

## 8. Running

Offline Q/V warmup:
```bash
VALUE_STEPS=20000 FULL_STEPS=10000 BATCH_SIZE=128 \
DEVICE=cuda:1 \
INIT_CHECKPOINT=<flow_ckpt.pt> \
LPB_CKPT=<bce_head.pth> \
bash robosuite/pipeline/scripts/utils/init_iql_qv.sh
```

Use the warmup checkpoint for online RL:
```bash
IQL_WARMUP_CKPT=<path/to/iql_state.pt> \
INIT_CHECKPOINT=<flow_ckpt.pt> \
LPB_CKPT=<bce_head.pth> \
bash robosuite/pipeline/scripts/train_dipole_rl.sh
```

Headless smoke:
```bash
INTERACTIVE=false VIEWER_ENABLED=false INTERVENTION_ENABLED=false \
INIT_CHECKPOINT=<flow_ckpt.pt> \
LPB_CKPT=<bce_head.pth> \
bash robosuite/pipeline/scripts/train_dipole_rl.sh \
  runtime.max_steps=500 \
  algorithm.trainer.pretrain_steps=20 \
  algorithm.q_learning.warmup_value_steps=20 \
  algorithm.q_learning.warmup_full_steps=10
```

Expected metrics after online updates:
- `flow_loss`
- `iql/q_loss`, `iql/v_loss`
- `disc/disc_loss`, `disc/threshold` when failure/non-failure pools are ready
- `G_mean`, `advantage_mean`, `disc_reward_mean`

## 9. Verification

Useful checks:
```bash
/home/dodo/miniconda3/envs/dagger/bin/python -m py_compile \
  robosuite/pipeline/algorithms/q_learning/*.py \
  robosuite/pipeline/algorithms/discriminator/*.py \
  robosuite/pipeline/algorithms/dipole/*.py

/home/dodo/miniconda3/envs/dagger/bin/python -m pytest \
  robosuite/pipeline/algorithms/q_learning/tests \
  robosuite/pipeline/algorithms/discriminator/tests \
  robosuite/pipeline/algorithms/dipole/tests
```

Regression checks:
- `g_mode=bce_frozen` should avoid AdvantageG and preserve legacy DIPOLE behavior.
- Encoder identity should be shared across trainer, IQL replay, discriminator, and AdvantageG.
- `disc_reward_mean` should be in `(-1, 0)` when discriminator reward is active.
- A new Q-ensemble warmup checkpoint should load cleanly; old two-Q checkpoints should fail fast.

## 10. Implementation History Summary

This section replaces the old per-subagent log. It keeps only the information future coding agents need.

Completed implementation groups:
- Shared encoder: introduced `SharedFrozenEncoder`, camera binding, frozen LPB v2 encoder reuse, and action-aware latent encoding.
- IQL: added Q-chunk `IQLLearner`, replay wrapper, Q/V warmup, and later upgraded to Q ensemble with `K=5`, `M=2`.
- Online discriminator: added single-frame `OnlineBCEDiscriminator`, failure/non-failure replay, LPB warm-start, and reward formula `-sigmoid(failure_score - tau)`.
- AdvantageG: added `AdvantageGProvider` with `alpha * norm(advantage) + beta * norm(-disc_logit)`.
- Integrated trainer: wired shared encoder, IQL, discriminator, replay fanout, warmup loading, advantage provider switch, and async learner compatibility.
- Warmup CLI: added Hydra warmup entry and `init_iql_qv.sh`; latest path supports LPB reward annotations and preencoded replay cache.
- Bug fixes: corrected discriminator label convention, removed obsolete reward sign knobs, aligned BCE head to single-frame LPB v2, ensured checkpoints include debug IQL/discriminator state, and made `dipole_rl` factory/config paths usable.
- Performance work: added offline preencoding cache for Q/V warmup. Online encoder caching remains open.

Removed or obsolete ideas:
- Chunk-wise discriminator head with `D_ctx + H * D_a` input.
- `disc_reward_sign` and `disc_reward_source` config knobs.
- Re-negating discriminator rewards in IQL replay.
- Loading old two-Q IQL checkpoints into the ensemble learner.
- Using AdvantageG for single-observation display.

## 11. Known Risks And Open Work

- Online learner throughput is still limited by repeated encoder forwards across IQL replay, discriminator replay, and AdvantageG. A short-lived per-train-step context cache keyed by storage index is the next likely speedup.
- The learner thread remains single-process and single-device by design. Do not introduce DDP or multiprocessing in the online learner without changing the architecture contract.
- Discriminator updates require both failure and non-failure samples. With no interventions, `disc/*` metrics may be absent because the disc update is correctly skipped.
- Checkpoint resume for attached `iql_state` and `discriminator_state` is not a full training resume path yet; treat those fields as debug snapshots unless explicit resume support is added.
- Offline warmup cache can be stored on CPU by default. Moving it to learner GPU can speed sampling but may consume too much memory for large demo sets.
