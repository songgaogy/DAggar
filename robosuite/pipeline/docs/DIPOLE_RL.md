# DIPOLE-RL — DIPOLE + IQL Q-chunking + Online BCE Discriminator

This document is the design spec for extending DIPOLE (currently CFG-style polarity flow with a **frozen** LPB v2 BCE detector) into a fully online human-in-the-loop RL pipeline.

The DIPOLE algorithm itself (steps 1–4 of the project plan) is already in place — see `DIPOLE.md`. This document specifies steps **(5) offline Q/V warmup** and **(6) online co-training of flow policy + Q/V + discriminator**.

---

## 1. What changes vs current DIPOLE

| Aspect | Current DIPOLE (`g_mode=bce_frozen`) | DIPOLE-RL (`g_mode=advantage`) |
|---|---|---|
| G signal | `G = -raw_bce_score` (frozen BCE) | `G = α·A(s,a) + β·(-disc_logit)` |
| Advantage | (none) | `A = min(Q1, Q2) − V`, Q-chunking |
| Discriminator | frozen lpb_v2 BCE head | **online-trainable** BCE head; encoder still frozen |
| Reward | (unused) | `r_total = r_env + λ_disc · r_disc`; Q/V learn from it |
| Encoder | per-provider, embedded in `g_provider.py` | **single** `SharedFrozenEncoder` shared across DIPOLE / IQL / disc |
| GPU layout | inference cuda:0, learner cuda:1 | unchanged; learner thread now runs **3 losses** per tick |
| Concurrency | 1 daemon learner thread | unchanged (still 1 daemon learner thread) |

The DIPOLE flow policy code (`models/flow.py`) is unchanged. The only hook is `agent.attach_g_provider(...)`, which now receives `AdvantageGProvider` instead of `LPBV2GProvider`.

---

## 2. Architecture

```txt
┌─────────────────────── cuda:1 (learner GPU) ─────────────────────────┐
│                                                                      │
│   SharedFrozenEncoder  ←─── owned by trainer, no_grad always         │
│        │                                                             │
│        ├─ context for IQL Q/V       (chunk-centric step_batch)       │
│        ├─ context for online BCE    (balanced expert/policy batch)   │
│        └─ context for AdvantageG    (actor_batch on DipoleBatch)     │
│                                                                      │
│   IQL learner (Q1, Q2, V, target_V)  ─────┐                          │
│   OnlineBCEDiscriminator (head only)   ──┐│                          │
│                                          ││                          │
│        ┌─────────────────────────────────┤├─────────────┐            │
│        │  AdvantageGProvider             ↓              │            │
│        │  G = α·A_norm + β·(−disc_logit_norm)           │            │
│        └────────────────────────┬─────────────────────  │            │
│                                 │                                    │
│   DIPOLE flow policy training (polarity dual-branch loss)            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────── cuda:0 (inference GPU) ───────────────────────┐
│   Inference copy of DIPOLE flow policy (shadow model)                │
│   Frozen encoder (separate copy on cuda:0 for eval-time G)           │
└──────────────────────────────────────────────────────────────────────┘
```

### Per-tick step order (single learner thread, cuda:1)

1. Sample `step_batch` (chunk + reward + next_obs + done) and `actor_batch` (raw chunk + context) from the shared replay store.
2. `no_grad`: re-evaluate `r_disc` on `step_batch` with the **current** disc to avoid stale rewards; overwrite `step_batch.rewards`.
3. IQL Q loss: Bellman MSE for both critics.
4. IQL V loss: expectile(`min(Q1, Q2) − V`).
5. `no_grad`: compute `A` on `actor_batch` and `disc_logit` on its `(context, action_chunk_raw)`; normalize each; combine into G.
6. DIPOLE flow update with G (replaces existing `LPBV2GProvider.compute_g_for_batch(...)` path).
7. Sample balanced expert/policy batch from disc replay; one BCE step.
8. Polyak target V update.
9. Emit metrics; return to async queue.

Main thread: rollout → env step → intervention routing → `trainer.record_transition` (which fans out into DIPOLE buffer, IQL buffer, and disc buffer) → optional `sync_inference_policy`.

### Reward composition

```
r_total(s, a) = r_env(s, a) + λ_disc · r_disc(s, a)
              = (−1 step / 0 success) + λ_disc · (−disc_logit_normalized)
```

`r_total` is what IQL bootstraps off. The discriminator term is **always** computed under no_grad and uses the current discriminator weights (re-eval on sampled batch, not stored stale).

### G composition

```
A(s, a)        = min(Q1(s, a), Q2(s, a)) − V(s)          [IQL learner, no_grad]
disc_logit     = OnlineBCEDiscriminator.score(s, a).logit [no_grad]
A_norm         = batch_zscore(A)
disc_norm      = batch_zscore(disc_logit)
G              = α · A_norm + β · (−disc_norm)
```

DIPOLE then maps G to polarity weights `w_pos = sigmoid(β_dipole · G + k)` exactly as today.

---

## 3. Module layout

```
robosuite/pipeline/algorithms/
├── dipole/                                 # existing; minor extensions
│   ├── ... (unchanged)
│   ├── advantage_g_provider.py             # NEW
│   └── trainer.py                          # extended hook points only
├── q_learning/                             # NEW (IQL + Q-chunking)
│   ├── __init__.py
│   ├── common.py        # IQLConfig, IQLStepBatch, IQLActorBatch
│   ├── networks.py      # QChunkNetwork, VNetwork
│   ├── losses.py        # bellman_q_loss, expectile_v_loss, compute_advantage
│   ├── replay.py        # IQLReplayBuffer
│   ├── iql.py           # IQLLearner
│   ├── warmup.py        # offline warmup CLI
│   └── data_util.py     # chunk reward / done / proprio-as-action helpers
└── discriminator/                          # NEW
    ├── __init__.py
    ├── adapter.py       # placeholder for lpb_v2 re-exports
    ├── base.py          # DiscriminatorBase + dataclasses
    ├── bce_head.py      # TrainableBCEHead (with warm_start)
    ├── online_bce.py    # OnlineBCEDiscriminator + DiscriminatorConfig
    ├── replay.py        # DiscriminatorReplayBuffer (expert/policy balance)
    ├── encoder.py       # SharedFrozenEncoder
    └── losses.py        # bce_with_logits_loss (with label smoothing)
```

Entry points:
- `robosuite/pipeline/train_dipole_rl.py`
- `robosuite/pipeline/config/train_dipole_rl.yaml`
- `robosuite/pipeline/scripts/train_dipole_rl.sh`

---

## 4. Cross-module API contract

| Caller | Callee | Input shapes | Output shapes |
|---|---|---|---|
| IQL replay → encoder | `SharedFrozenEncoder.encode(image_obs_raw, proprio_raw)` | `(B, V, 3, H, W) ∈ [0,1]`, `(B, D_s)` | `(B, D_ctx)` |
| IQL replay → disc | `disc.intrinsic_reward(context, action_chunk)` | `(B, D_ctx)`, `(B, H, D_a)` | `(B,)` |
| Trainer → IQL | `iql.update(step_batch)` | `IQLStepBatch` | `dict[str, float]` |
| Trainer → IQL (warmup) | `iql.warmup_value_only(step_batch)` | `IQLStepBatch` | `dict[str, float]` |
| AdvantageG → IQL | `iql.compute_advantage_for_batch(actor_batch)` | `IQLActorBatch` | `(B,)` |
| AdvantageG → disc | `disc.score(context, action_chunk).logit` | `(B, D_ctx)`, `(B, H, D_a)` | `(B,)` |
| Flow update → G provider | `g_provider.compute_g_for_batch(dipole_batch)` | `DipoleBatch` | `(B,)` |
| Trainer → disc | `disc.update(disc_batch)` | `DiscriminatorBatch` | `dict[str, float]` |
| Main → trainer | `trainer.record_transition(...)` | existing signature | `Transition` |

### Hard prohibitions (all subagents)

- **No encoder ownership inside IQL/disc**: encoder is always injected.
- **Encoder must stay frozen**: no `encoder.train()`, no gradients through encoder. Any forward must be inside `torch.no_grad()`.
- **Q-chunk input shape is fixed**: `D_ctx + H·D_a` for Q; `D_ctx` for V. No per-step Q factorization.
- **Concurrency is fixed**: one daemon learner thread on cuda:1; no multiprocessing, no DDP.
- **`DipoleFlowPolicy.update()` signature must not change**: G enters exclusively via `compute_g_for_batch(batch) → (B,)`.
- **Markdown lives only under `pipeline/docs/`**; no new READMEs anywhere else.

---

## 5. Config (`config/train_dipole_rl.yaml`)

The RL config inherits `train_dipole.yaml` and adds:

```yaml
algorithm:
  type: "dipole_rl"
  dipole:
    g_mode: "advantage"               # "advantage" | "bce_frozen"
  q_learning:
    enabled: true
    config: {IQLConfig fields}
    warmup_value_steps: 20000
    warmup_full_steps: 5000
    warmup_ckpt: null                 # if set, skip warmup
  discriminator:
    enabled: true
    online_train: true
    warm_start_ckpt: "checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth"
    config: {DiscriminatorConfig fields}
  advantage_g_provider:
    alpha: 1.0
    beta: 0.5
    advantage_normalization: "batch_zscore"
    disc_normalization: "batch_zscore"
runtime:
  init_checkpoint: null
  bootstrap_g_with_frozen_bce: true   # during warmup, force bce_frozen
```

`DipoleConfig` gets a new field `g_mode: Literal["bce_frozen", "advantage"] = "bce_frozen"`. Default preserves backward compatibility for any existing DIPOLE run.

---

## 6. Subagent execution plan

Each prompt is fully self-contained; see `docs/prompts/`.

```
03_shared_encoder        (no deps; foundation)
       ↓
01_q_learning_iql   ──┐
02_online_disc      ──┤  (parallel; both depend on 03)
                      ↓
04_advantage_g_provider  (depends on 01 + 02 + 03)
                      ↓
05_integrated_trainer    (depends on 04 + existing DIPOLE trainer)
                      ↓
06_warmup_script         (depends on 05)
                      ↓
07_verification          (end-to-end smoke + regression check)
```

---

## 7. Verification (handled by `07_verification.md`)

1. `python -m py_compile` every new file — must parse.
2. Encoder identity check: `id(trainer.encoder) is id(iql.replay._encoder_ref) is id(disc._encoder_ref)`.
3. IQL warmup smoke: 200 steps, no NaN, V loss monotonic-ish down.
4. Headless integrated smoke: 500 env steps, metrics dict contains `flow_loss / q_loss / v_loss / disc_loss / G_mean / advantage_mean / disc_reward_mean`, no NaN/Inf.
5. `nvidia-smi`: cuda:0 has inference process; cuda:1 has training process; each < 20 GB.
6. Regression: `algorithm.dipole.g_mode=bce_frozen` reproduces existing DIPOLE behavior bit-for-bit on a fixed seed (or as close as fp64-vs-fp16 noise allows).

---

## 8. Known risks (carry forward)

- **Q-chunking with off-policy actions** is standard IQL; we evaluate historical-policy chunks. Document in `01_q_learning_iql`.
- **Single learner thread throughput**: if fps drops below 10, fall back to `disc.cfg.update_every_n_steps > 1`.
- **Discriminator cold start**: gate the advantage path via `runtime.bootstrap_g_with_frozen_bce` until warmup completes.
- **Intervention scarcity**: balanced sampler uses replacement on the expert pool.

See `DIPOLE.md` for the original CFG-style polarity weighting details (unchanged) and `README.md` for the entry-point commands.

---

## 9. For subagents

When you finish implementing your part, please update THIS FILE [here](#10-subagent-status)

---

## 10. Subagent status

### Subagent-3 · SharedFrozenEncoder — DONE (2026-05-20)

**Files touched**
- `robosuite/pipeline/algorithms/discriminator/encoder.py` — filled in `SharedFrozenEncoder.__init__` / `bind_policy_cameras` / `encode`. Public surface (consume from downstream subagents):
  - `inner_encoder: LPBV2Encoder` — the wrapped lpb_v2 encoder
  - `context_dim: int`, `action_input_dim: int`, `proprio_input_dim: int`, `view_names: list[str]`, `original_img_size: tuple[int, int]`, `frameskip: int`, `action_dim_per_step: int`, `feature_source: str`, `transformer_layer: int`, `device: str`, `bce_ckpt_path: str`
  - `encode(*, image_obs_raw=(B,V,3,H,W), proprio_raw=(B,D_s)) -> (B,D_ctx)` — runs degraded `f(o,s)` with `a := tile(s)`; always under `no_grad`.
  - `bind_policy_cameras(policy_cameras)` — idempotent on same tuple, hard-raises on a different tuple.
- `robosuite/pipeline/algorithms/dipole/g_provider.py` — `LPBV2GProvider` gains `shared_encoder: SharedFrozenEncoder | None = None`. When supplied, the provider reuses `shared_encoder.inner_encoder` and pulls metadata from it (single encoder copy on device). When omitted the legacy `bce_frozen` path (used by `train_dipole.py:689`) is unchanged.
- `robosuite/pipeline/algorithms/discriminator/tests/test_shared_encoder.py` + `tests/__init__.py` — pytest covers metadata, `requires_grad=False` on every encoder parameter, encode output shape `(B, D_ctx)`, missing view → `KeyError`, double-bind mismatch → `RuntimeError`. Skips when no CUDA or no BCE ckpt; honors `DIPOLE_BCE_CKPT` env var, else globs the newest `checkpoints/lpb_v2/bce_eval_robosuite/run_*/checkpoints/bce_head.pth`.

**Verified**
- `py_compile` clean on both edited modules.
- 6/6 unit tests pass on a CUDA host against `checkpoints/lpb_v2/bce_eval_robosuite/run_20260516_202016/checkpoints/bce_head.pth` (auto-resolves to dynamics `model_49.pth`).
- Identity check: `LPBV2GProvider(..., shared_encoder=enc).encoder is enc.inner_encoder` holds; the legacy `shared_encoder=None` path builds an independent encoder copy.

**Notes for subagents 01 / 02 / 04**
- Build exactly one `SharedFrozenEncoder` in the trainer and inject it into IQL, OnlineBCE, and (optionally) `LPBV2GProvider`.
- Treat `enc.inner_encoder` as the upstream `LPBV2Encoder` if you need the raw `encode_batch(images, proprio, actions_real)` signal (e.g. the legacy BCE path); use `enc.encode(...)` everywhere the degraded action-independent `f(o, s)` is required.
- `enc.bind_policy_cameras(agent.camera_names)` must be called once during trainer init, before any consumer calls `encode`.
- `q_learning/data_util.py::tile_proprio_as_action` is still a stub. The shared encoder inlines the tile via the private `_tile_proprio_as_action` helper in `encoder.py`. When subagent-01 implements the public function, switch the encoder over to it for a single source of truth.


### Subagent-1 · IQL Q-chunking — DONE (2026-05-21)

**Files touched**
- `robosuite/pipeline/algorithms/q_learning/common.py` — `IQLStepBatch.to` / `IQLActorBatch.to` (tensor-only device moves, metadata kept by reference).
- `robosuite/pipeline/algorithms/q_learning/losses.py` — `bellman_q_loss` (MSE), `expectile_v_loss(diff, tau)` (asymmetric L2), `compute_advantage` (returns shape `(B,)` after squeeze).
- `robosuite/pipeline/algorithms/q_learning/data_util.py` — `aggregate_chunk_reward` (Σ γⁱ rᵢ), `chunk_done_mask` (any-done → (B,1)), `tile_proprio_as_action` (repeat-pad to `action_input_dim`). Duplicates the encoder's private helper; consolidation TODO (see §10 note above).
- `robosuite/pipeline/algorithms/q_learning/networks.py` — `_build_mlp` with `Linear → LayerNorm → GELU` blocks, Kaiming-normal hidden init, **zero-init final layer (weight + bias)**. `QChunkNetwork` input dim `D_ctx + H · D_a_policy`; `VNetwork` input dim `D_ctx`.
- `robosuite/pipeline/algorithms/q_learning/replay.py` — `IQLReplayBuffer` wraps any `FlowDaggerReplayBuffer`-style store by reference. Reuses `_storage / _lock / _get_valid_start_indices_locked`. Adds explicit `start + H` boundary safety: falls back to `_storage[start+H-1].next_obs` and forces `done=1` when the next index is out-of-storage or cross-episode. Disc reward is optional — `None` → effective_disc_coef=0 (cfg untouched). Disc reward is sampled ONCE per chunk and divided by H before being broadcast across the chunk.
- `robosuite/pipeline/algorithms/q_learning/iql.py` — `IQLLearner`. Two optimizers (`AdamW(q1+q2, lr=q_lr)`, `AdamW(v, lr=v_lr)`), target_v cloned + frozen. `update` does Bellman Q (sum over both critics, then Huber-free MSE), expectile V on detached `min(Q1, Q2) - V`, then polyak on V only. `warmup_value_only` regresses V toward `r + γ^H·(1-done)·target_v(s')` (MSE) and still polyaks. `state_dict` packs `{q1, q2, v, target_v, q_optim, v_optim, cfg, context_dim, action_dim}` and `load_state_dict(strict=True)` hard-asserts on `context_dim` and `action_dim` mismatch.
- `robosuite/pipeline/algorithms/q_learning/warmup.py` — Hydra-driven CLI on `config/train_dipole_rl.yaml`. Builds a robosuite env (for the `RobosuiteProprioExtractor`), loads HDF5 expert/success/fail splits via `resolve_task_demo_paths` + `load_hdf5_demos_into_flow_transitions`, annotates `info` with `episode_index / episode_step / episode_namespace / buffer_role="offline"` so the chunk-window valid-start cache bounds chunks per demo, then runs `warmup_value_only` then `update`. Disc is passed `None` (warmup is BCE-free until subagent-2 lands; r_disc = 0). Closes the env after data loading. Output schema:
  ```python
  {
      "iql_state": iql.state_dict(),     # see iql.py
      "cfg": asdict(IQLConfig),
      "encoder_meta": {
          "bce_ckpt": str,
          "context_dim": int,
          "view_names": list[str],
          "policy_camera_names": list[str],
          "task": str,                   # data task name (e.g. PandaPickPlaceBread)
          "task_env": str,               # env name (e.g. PickPlaceBread)
          "policy_action_dim": int,
      },
      "schema_version": 1,
  }
  ```
- `robosuite/pipeline/algorithms/q_learning/tests/test_iql.py` (+ `tests/__init__.py`) — 10 CPU-only unit tests covering `compute_advantage`, `expectile_v_loss` (tau=0.5 sanity, asymmetry), `bellman_q_loss`, `aggregate_chunk_reward`, `chunk_done_mask`, one `update` step (V move guarded by a prior `warmup_value_only` to escape the zero-init dead-point), one `warmup_value_only` step (Q must NOT move), `state_dict / load_state_dict` round-trip, and a context-dim mismatch hard-assert. **All 10 pass.**
- `robosuite/pipeline/scripts/init_iql_qv.sh` — bash entry mirroring `train_dipole.sh` env-var conventions (`ENVIRONMENT`, `INIT_CHECKPOINT`, `LPB_CKPT`, `VALUE_STEPS`, `FULL_STEPS`, `BATCH_SIZE`, `DEVICE`, `OUTPUT_DIR`, `NUM_TRAJECTORIES`). Writes `outputs/DIPOLE_RL/iql_qv_cache/${ENVIRONMENT}/iql_state.pt`. Headless by default (`MUJOCO_GL=egl`).

**Verified**
- `python -m py_compile` clean on all `q_learning/*.py`.
- `pytest robosuite/pipeline/algorithms/q_learning/tests/test_iql.py -v` → 10/10 pass (CPU only).

**Notes for subagent-04 (AdvantageGProvider)**
- `iql.compute_advantage_for_batch(actor_batch) -> (B,)` (already `.squeeze(-1)`-ed by `compute_advantage`). Consumers should z-score directly — no need to flatten.
- `IQLActorBatch.action_chunk_raw` is **un-normalized**; this is what the Q networks are trained on (the replay buffer feeds raw actions in the step batch too, since policy normalization stats are not applied inside IQL).

**Notes for subagent-05 (integrated trainer / `train_dipole_rl.py`)**
- Load warmup ckpt with:
  ```python
  ckpt = torch.load(cfg.algorithm.q_learning.warmup_ckpt, map_location="cpu",
                    weights_only=False)
  iql.load_state_dict(ckpt["iql_state"], strict=True)
  assert ckpt["encoder_meta"]["context_dim"] == encoder.context_dim
  assert ckpt["encoder_meta"]["policy_action_dim"] == policy_action_dim
  ```
- When wiring the online learner thread, build `IQLReplayBuffer` around the trainer's online buffer (a `DipoleReplayBuffer` / `FlowDaggerReplayBuffer`). Tag every online transition's `info` with `buffer_role="online"` and `episode_index / episode_step / episode_namespace` if you want `IQLStepBatch.is_online` to be meaningful; offline-loaded demos carry `buffer_role="offline"` already.
- Pass the live `OnlineBCEDiscriminator` (once subagent-2 lands) to `sample_step_batch(..., discriminator=disc)`; it is invoked under `no_grad` per sample — no stale rewards.

**Notes for subagent-02 (OnlineBCEDiscriminator) — contract used here**
- `discriminator.intrinsic_reward(context=(B, D_ctx), action_chunk=(B, H, D_a)) -> (B,)` is the only method the IQL replay calls during sampling. Sign convention: higher = more expert-like; IQL applies this with `cfg.disc_reward_coef` as-is (no extra negation inside replay), matching the docs §B reward composition.


### Subagent-2 · OnlineBCEDiscriminator — DONE (2026-05-21)

**Files touched**
- `robosuite/pipeline/algorithms/discriminator/bce_head.py` — `TrainableBCEHead`: `[Linear → LayerNorm → GELU] × num_layers → Linear(_, 1)`, Kaiming on hidden, zero on final Linear. forward squeezes to `(B,)`. `warm_start_from_lpb_bce_ckpt` loads `state['bce_detector']['head']` with `strict=False` and asserts ≥ 1 tensor was actually overwritten (typically every layer except the first `Linear` because our head sees `D_ctx + H·D_a` while the lpb_v2 ckpt was sized `D_ctx`-only).
- `robosuite/pipeline/algorithms/discriminator/losses.py` — `bce_with_logits_loss(logits, labels, label_smoothing, pos_weight)` with `y' = y·(1−2s) + s`.
- `robosuite/pipeline/algorithms/discriminator/online_bce.py` — `OnlineBCEDiscriminator`:
  - `__init__` builds head + AdamW on cuda:1, optional warm_start.
  - `score(context, action_chunk)` (no_grad) → `DiscriminatorOutput` with `logit / prob_expert / decision (logit > threshold)` and metadata `{threshold, normalized_margin, prediction}` (mirrors the AWR baseline's `AWRDiscriminatorLabel` field set referenced in `prompts/02`).
  - `intrinsic_reward(...)` (no_grad) returns raw logit `(B,)` — higher = expert-like. IQL is the one that negates via `disc_reward_sign`.
  - `update(batch)` one optim step + EMA threshold from the median of the policy-half (label < 0.5) logits. Returns `{disc_loss, disc_acc, expert_logit_mean, policy_logit_mean, threshold}`.
  - `state_dict / load_state_dict` round-trips head weights + AdamW state + threshold + step + cfg; hard-asserts on `context_dim / action_dim / action_horizon` mismatch.
- `robosuite/pipeline/algorithms/discriminator/replay.py` — `DiscriminatorReplayBuffer` (IQL-style index-only wrapper). Holds two cached index lists `_expert_starts / _policy_starts` over the base buffer's `_get_valid_start_indices_locked()`. A chunk is "expert" iff **any** step in its H-window has `is_intervention=True` OR its first storage index ∈ `_demo_indices` (loose criterion, per user choice). `add_from_transition` is a no-op — the base buffer is the single source of truth. `bootstrap_from_demos` calls `base.add(t)` for each demo and marks the new indices as forced-expert. `sample(batch_size, device)` returns `DiscriminatorBatch` with frozen-encoder context.
- `robosuite/pipeline/algorithms/discriminator/tests/test_online_bce.py` — 9 CPU-only tests: head forward shape, warm-start (skipped without ckpt), score shape / no_grad, update loss-drop on a separable batch, intrinsic-reward sign, state_dict roundtrip, dim-mismatch hard-assert, BCE loss smoothing math, replay-buffer balance + bootstrap. **All 9 pass** under `pytest discriminator/tests/test_online_bce.py -v`.

**Verified**
- `py_compile` clean on all `discriminator/*.py`.
- `pytest robosuite/pipeline/algorithms/discriminator/tests/test_online_bce.py -v` → 9/9 pass on CPU.

**Notes for subagent-04 (AdvantageGProvider)**
- Use `disc.score(context=ctx, action_chunk=actor_batch.action_chunk_raw).logit` for the disc term in `G`. Sign convention: higher = expert-like, so `G = α·A_norm + β·(−disc_logit_norm)` per docs §2.
- The metadata dict on the `DiscriminatorOutput` exposes `normalized_margin` for any UI / logging needs.
- Threshold is EMA-tracked and is NOT used by AdvantageG — it's only consumed by `score(...).decision` for the optional UI signal.

**Notes for subagent-05 (integrated trainer / `train_dipole_rl.py`)**
- Construct in this order (encoder → disc → disc_replay):
  ```python
  shared_encoder = SharedFrozenEncoder(bce_ckpt, device="cuda:1")
  shared_encoder.bind_policy_cameras(agent.camera_names)
  disc = OnlineBCEDiscriminator(
      cfg=disc_cfg,
      encoder=shared_encoder,
      context_dim=shared_encoder.context_dim,
      action_dim=policy_action_dim,
      action_horizon=dipole_cfg.action_horizon,
  )
  disc_replay = DiscriminatorReplayBuffer(
      disc_cfg, base_buffer=trainer.replay,
      encoder=shared_encoder, action_horizon=dipole_cfg.action_horizon,
  )
  ```
- Learner-thread tick:
  ```python
  if disc_replay.ready(disc_cfg.batch_size) and step % disc_cfg.update_every_n_steps == 0:
      metrics_disc = disc.update(disc_replay.sample(disc_cfg.batch_size, device="cuda:1"))
  ```
- Pass `disc` to `iql_replay.sample_step_batch(..., discriminator=disc)` so r_disc is re-evaluated under no_grad per sample (already wired in Subagent-1's replay; just lift the `None` default).
- `bootstrap_g_with_frozen_bce`: while it's true, skip `disc_replay.sample` / `disc.update` entirely and feed G via the legacy `LPBV2GProvider` path.
- The `warm_start_ckpt` config knob points at `checkpoints/lpb_v2/bce_viz_robosuite/<run_name>/checkpoints/bce_head.pth` per the user's note in `prompts/02`.

**Notes for subagent-01 (IQL) — already satisfied**
- The `disc=None` default in `sample_step_batch` matches the warmup contract: when subagent-02 isn't wired yet, no behaviour changes.
- `intrinsic_reward` returns the raw `(B,)` logit; IQL's `disc_reward_sign="negate_logit"` does the final flip.


### Subagent-4 · AdvantageGProvider — DONE (2026-05-21)

**Files touched**
- `robosuite/pipeline/algorithms/dipole/advantage_g_provider.py` — filled in the `__init__`, `compute_g_for_batch`, `_normalize`, and `compute_g_for_observation` bodies. Mixing math follows docs §2:
  ```
  context     = encoder.encode(image_obs_raw, proprio_raw)         # (B, D_ctx)
  advantage   = iql.compute_advantage_for_batch(                   # (B,)
                    IQLActorBatch(context, action_sequences_raw))
  disc_logit  = disc.score(context, action_sequences_raw).logit    # (B,)
  G           = α · z(advantage) + β · (−z(disc_logit))            # (B,)
  ```
  All four normalization modes from `DipoleFlowPolicy._normalize_g` (`none / batch_zscore / running_zscore / minmax`) are mirrored. `running_zscore` keeps separate EMA buffers per channel (`_adv_running_*` vs `_disc_running_*`) with momentum=0.1 to match `flow.py:313-341`.
- `robosuite/pipeline/algorithms/dipole/tests/test_advantage_g_provider.py` + `tests/__init__.py` — 8 CPU-only tests with fake encoder / IQL / disc. Covers: β=0 → pure z(advantage); α=0 → −z(disc_logit); `none` mode → exact linear mixing; output shape `(B,)` no_grad; `compute_g_for_observation` raises NotImplementedError; `bind_policy_cameras` proxy; running_zscore EMA state advances over two calls; unknown normalization mode rejected at construction.

**Verified**
- `python -m py_compile` clean on the new module + tests.
- `pytest robosuite/pipeline/algorithms/dipole/tests/test_advantage_g_provider.py -v` → 8/8 pass on CPU.

**Notes for subagent-05 (integrated trainer / `train_dipole_rl.py`)**
- Construction order (after the encoder/disc/iql snippet from subagent-02's section):
  ```python
  adv_provider = AdvantageGProvider(
      iql_learner=iql,
      discriminator=disc,
      encoder=shared_encoder,
      alpha=cfg.algorithm.advantage_g_provider.alpha,
      beta=cfg.algorithm.advantage_g_provider.beta,
      advantage_normalization=cfg.algorithm.advantage_g_provider.advantage_normalization,
      disc_normalization=cfg.algorithm.advantage_g_provider.disc_normalization,
  )
  adv_provider.bind_policy_cameras(list(agent.camera_names))
  agent.attach_g_provider(adv_provider)
  ```
- `compute_g_for_observation(...)` **raises NotImplementedError**. If the RL trainer needs a single-observation BCE display (e.g. the live UI overlay in `train_dipole.py:1339`), keep the legacy `LPBV2GProvider` around in parallel and call it ONLY for the eval-time overlay; don't try to route through `AdvantageGProvider`.
- Encoder is forwarded per call — no caching. Subagent-05 should keep exactly one `SharedFrozenEncoder` instance and share it across IQL, disc, and this provider so the encoder forwards don't duplicate.
- During the `runtime.bootstrap_g_with_frozen_bce` window the trainer should keep `LPBV2GProvider` attached instead of swapping to AdvantageGProvider; switch only after the warmup budget is met.
- Running-EMA buffers (`_adv_running_*`, `_disc_running_*`) are private to this provider and reset on each new instance. If you want them in the trainer checkpoint, add a thin `state_dict / load_state_dict` pair (out of scope for subagent-04).
- `α / β` are stored as floats on the provider; expose them through the trainer's reload path if you want to schedule them over training.


### Subagent-5 · Integrated Trainer — DONE (2026-05-21)

**Files touched**
- `robosuite/pipeline/algorithms/dipole/common.py` — added `g_mode: str = "bce_frozen"` field to `DipoleConfig` (backward compatible default; `train_dipole.py` runs unchanged).
- `robosuite/pipeline/algorithms/dipole/agent.py` — `DipoleAgent.from_config` now reads `g_mode` (with a hard-assert that it is one of `{"bce_frozen", "advantage"}`); added `attach_iql_learner(learner)` and `attach_discriminator(disc)` setters mirroring `attach_g_provider` (both stash the reference on `self.core` for observability only — flow model unchanged).
- `robosuite/pipeline/algorithms/dipole/trainer.py` — `DipoleTrainer.__init__` gained `iql_learner / discriminator / iql_replay / disc_replay / shared_encoder / learner_device / iql_batch_size / disc_batch_size / disc_update_every_n_steps` kwargs (all default `None`/`64`/`1`, so the legacy DIPOLE trainer construction `DipoleTrainer(agent)` still works). `record_transition` now also dispatches to `disc_replay.add_from_transition` when set (a no-op today — disc reclassifies lazily at sample time). `train_step` is the fused per-tick path:
  ```
  flow (agent.update on demo_batch)
    └─> iql_learner.update(step_batch)  if iql_replay.ready(...)
         (and stamps metrics["iql/*"], advantage_mean, disc_reward_mean)
    └─> discriminator.update(disc_replay.sample(...))
         every `disc_update_every_n_steps` ticks (and stamps metrics["disc/*"])
  ```
  IQL's `update()` is a single call that does Q-loss, V-loss, and Polyak target update; the spec's "Q → V" sub-steps are inside it. New `pretrain_iql_value(num_steps)` and `pretrain_iql_full(num_steps)` helpers drive the warmup with `discriminator=None` (warmup is BCE-free per subagent-1's contract).
- `robosuite/pipeline/train_dipole_rl.py` — new Hydra entry point (1110 lines). Near-verbatim clone of `train_dipole.main`; RL additions tagged with `# RL-ADD` / `# RL-EDIT`. Key new sections, in order:
  1. `SharedFrozenEncoder` constructed on `cfg.algorithm.q_learning.config.device` from `cfg.algorithm.discriminator.warm_start_ckpt`; `bind_policy_cameras` once.
  2. `LPBV2GProvider` retained as `legacy_g_provider` (shares the same encoder copy via `shared_encoder=...`); used unconditionally for the discriminator-display block in the rollout loop (replaces line 1339 of `train_dipole.py`).
  3. `IQLLearner` + `IQLReplayBuffer` and `OnlineBCEDiscriminator` + `DiscriminatorReplayBuffer`, with a hard-assert that all of `iql_cfg.device == disc_cfg.device == shared_encoder.device == learner_device`.
  4. Trainer rebuilt with all RL hooks, then `agent.attach_g_provider(legacy_g_provider)` so the flow base-policy pretrain runs against frozen BCE G (matches `runtime.bootstrap_g_with_frozen_bce=true`).
  5. Offline demos are double-routed: `trainer.bootstrap_demo_buffer(demos)` for `demo_buffer` (flow loss) and `disc_replay.bootstrap_from_demos(annotated_demos)` for `agent.online_buffer` (IQL + disc; demos pre-stamped with `episode_index / episode_step / buffer_role="offline"` via the new private `_annotate_offline_demos` helper).
  6. IQL warmup: either `iql.load_state_dict(payload["iql_state"])` from `cfg.algorithm.q_learning.warmup_ckpt` (with `context_dim` and `policy_action_dim` hard-asserts against the saved `encoder_meta`), or in-process `trainer.pretrain_iql_value(warmup_value_steps)` followed by `trainer.pretrain_iql_full(warmup_full_steps)`.
  7. `g_mode` switch: if `"advantage"`, construct `AdvantageGProvider` and `agent.attach_g_provider(advantage_g)` (overrides the legacy_g_provider attached above); if `"bce_frozen"`, keep legacy_g_provider. `legacy_g_provider` is always kept around for the disc-display path so `compute_g_for_observation` never raises.
  8. Online transitions tagged with `info["buffer_role"]="online"` before `trainer.record_transition` so `IQLStepBatch.is_online` is meaningful.
- `robosuite/pipeline/scripts/train_dipole_rl.sh` — replaced placeholder echo with the real Hydra invocation forwarding `INIT_CHECKPOINT`, `LPB_CKPT`, `IQL_WARMUP_CKPT`, `ALPHA`, `BETA`, `G_MODE`, `INTERACTIVE`, `VIEWER_ENABLED`, `INTERVENTION_ENABLED`. Extra CLI args are forwarded via `"$@"`.

**Hard contracts honored**
- Single daemon learner thread: `_async_update_loop` calls the new fused `train_step` once per tick. No multiprocessing, no DDP.
- Step order: flow → IQL (Q+V+polyak inside one `iql.update`) → disc. Each sub-step is skipped iff its component is `None`.
- `DipoleFlowPolicy.update()` signature unchanged. G provider is the only injection point.
- Encoder is owned by the trainer (not by IQL or disc); both modules receive it via constructor injection; `bind_policy_cameras` is called once immediately after construction.
- Encoder stays frozen — `OnlineBCEDiscriminator.update()` only optimizes the head; IQL never touches the encoder; `AdvantageGProvider` uses `encoder.encode(...)` under no_grad.
- `train_dipole.py` is **not modified** — only sibling imports.
- All docs live under `pipeline/docs/`.

**Verified**
- `python -m py_compile` clean on all 7 touched python files (`common.py`, `agent.py`, `trainer.py`, `advantage_g_provider.py`, `online_bce.py`, `iql.py`, `train_dipole_rl.py`).
- Module import works: `from robosuite.pipeline import train_dipole_rl; train_dipole_rl.main`.
- Existing CPU unit tests still pass: 33/33 across `q_learning/tests/`, `discriminator/tests/`, `dipole/tests/`.
- `DipoleConfig().g_mode == "bce_frozen"` by default → `train_dipole.py` is bit-compatible with the prior behavior.
- `bash -n scripts/train_dipole_rl.sh` clean.

**Notes for subagent-6 (warmup script) / subagent-7 (verification)**
- The integrated trainer is the consumer of `q_learning/warmup.py`'s ckpt; loader is `train_dipole_rl._load_iql_warmup_state`. It expects `payload["iql_state"]`, `payload["encoder_meta"]["context_dim"]`, and `payload["encoder_meta"]["policy_action_dim"]` — already produced by subagent-1.
- Headless smoke target (subagent-7):
  ```bash
  INTERACTIVE=false VIEWER_ENABLED=false INTERVENTION_ENABLED=false \
      INIT_CHECKPOINT=<path> bash robosuite/pipeline/scripts/train_dipole_rl.sh \
      runtime.max_steps=500 algorithm.trainer.pretrain_steps=20 \
      algorithm.q_learning.warmup_value_steps=20 \
      algorithm.q_learning.warmup_full_steps=10
  ```
  Expected metrics keys after first online tick: `flow_loss / iql/q_loss / iql/v_loss / disc/disc_loss / G_mean / advantage_mean / disc_reward_mean`.
- Regression smoke: same script with `algorithm.dipole.g_mode=bce_frozen` reproduces the legacy DIPOLE trainer because the AdvantageGProvider is never attached (the legacy LPBV2GProvider stays bound).
- Per-tick GPU work tripled vs legacy DIPOLE. If FPS drops below 10, raise `cfg.algorithm.discriminator.config.update_every_n_steps` (yaml default is 1) — the trainer's `disc_update_every_n_steps` flows from it directly.


### Subagent-6 · Offline Q/V Warmup CLI — DONE (2026-05-21)

**Files touched**
- `robosuite/pipeline/algorithms/q_learning/warmup.py` — Subagent-1 already shipped the Hydra warmup CLI with `discriminator=None` (BCE-free warmup, valid before subagent-2 landed). Subagent-6 attaches a **frozen** `OnlineBCEDiscriminator` built from `cfg.algorithm.discriminator.config` and warm-started from `algorithm.discriminator.warm_start_ckpt` (resolved via `to_absolute_path`). Construction order: encoder → IQL learner → replay → disc; device consistency hard-asserted (`disc_cfg.device == iql_cfg.device == encoder.device`). The head is `eval()`-ed, every parameter is `requires_grad_(False)`-flipped, and `disc.update()` is **never** invoked — only `intrinsic_reward(...)` under `no_grad` through `IQLReplayBuffer.sample_step_batch(..., discriminator=disc)`. Both warmup loops (`warmup_value_only` and `update`) now pass `discriminator=discriminator`. `payload["encoder_meta"]` gains two cross-check fields: `disc_warm_start_ckpt` and `disc_reward_coef` (mirrors the IQLConfig field used during warmup).
- `robosuite/pipeline/algorithms/dipole/trainer.py` — `pretrain_iql_value` and `pretrain_iql_full` now forward `self.discriminator` instead of hard-coding `None`. The trainer already owns the live disc reference; warmup just plumbs it through the same `sample_step_batch` path. Docstrings updated.

**Why it changed**
- prompt-06 §4 mandates a frozen disc during warmup so the reward composition `r_total = r_env + disc_reward_coef · r_disc` matches the online phase.
- Otherwise V/Q would be calibrated against `r_disc = 0` and would jump on the first online tick when the disc contribution suddenly turns on.
- The trainer-side change keeps the two warmup entry points (standalone CLI in `q_learning/warmup.py` and in-process from `train_dipole_rl.py:920+`) behaviorally identical.

**Hard contracts honored**
- Disc head stays frozen for the entire warmup (`eval()` + `requires_grad_=False` + no `.update()`). Encoder stays frozen as before.
- No new files; no new docs outside `pipeline/docs/`.
- Hydra-driven CLI preserved (matches the rest of the pipeline and `init_iql_qv.sh`); argparse rewrite intentionally skipped per user decision, since the existing wrapper script already provides the "lean CLI" feel via env vars.
- `IQLReplayBuffer.sample_step_batch` signature unchanged.

**Verified**
- `python -m py_compile robosuite/pipeline/algorithms/q_learning/warmup.py robosuite/pipeline/algorithms/dipole/trainer.py` → clean.
- `pytest robosuite/pipeline/algorithms/q_learning/tests/ robosuite/pipeline/algorithms/discriminator/tests/ robosuite/pipeline/algorithms/dipole/tests/ -v` → **33/33 pass** (no regressions from the new wiring).

**Notes for subagent-7 (verification)**
- The saved `iql_state.pt` now carries `payload["encoder_meta"]["disc_warm_start_ckpt"]` and `payload["encoder_meta"]["disc_reward_coef"]`. On the online path, `train_dipole_rl._load_iql_warmup_state` can cross-check these against the live `disc_cfg.warm_start_ckpt` / `iql_cfg.disc_reward_coef` to guard against silent config drift between warmup and online runs.
- End-to-end smoke target (substitute real ckpt paths):
  ```bash
  VALUE_STEPS=20 FULL_STEPS=10 BATCH_SIZE=16 NUM_TRAJECTORIES=2 \
  INIT_CHECKPOINT=<flow ckpt> LPB_CKPT=<bce_head.pth> \
      bash robosuite/pipeline/scripts/init_iql_qv.sh
  ```
  Expected stdout fingerprint: `[warmup] frozen disc ready (OnlineBCEDiscriminator); ...` before the value/full loop logs; non-empty `iql_state.pt` afterward.
- Disc influence sanity check: re-run warmup with `algorithm.q_learning.config.disc_reward_coef=0` and compare `target_mean` between the two runs — they must differ, confirming r_disc actually enters the warmup reward.


### Subagent-7 · End-to-end verification + bug sweep — DONE (2026-05-21)

Full pass/fail per Step 1–6 lives in `outputs/DIPOLE-RL_verification.md`. Summary below for downstream subagents.

**Result**: all six verification steps pass. Five real bugs fixed in flight (none of them blockers, but all of them silent footguns).

**Bug #1 — discriminator labeling was inverted** *(user-flagged)*
- `DiscriminatorReplayBuffer` was tagging offline expert demos as `label=1` ("expert") against a warm-started LPB v2 BCE head that is semantically a **failure** detector (high logit = failure-like). This made `r_disc = -logit` *penalize* expert behavior the moment human intervention data joined the buffer.
- Fix (this report):
  - Demos with `is_intervention=False` → `label=0` (non-failure pool).
  - Intervention chunks → `label=1` (failure pool).
  - Renamed `_expert_starts/_policy_starts` → `_failure_starts/_non_failure_starts`, `num_expert/num_policy` → `num_failure/num_non_failure`.
  - `DiscriminatorOutput.prob_expert` → `prob_failure`.
  - `update` metrics: `expert_logit_mean` → `failure_logit_mean`, `policy_logit_mean` → `non_failure_logit_mean`.
- `AdvantageGProvider`'s `G = α·z(A) + β·(−z(disc_logit))` sign is unchanged — `−disc_logit` already pointed in the correct direction *for the failure-detector convention*; the labels were what was pointing the wrong way.

**Bug #2 — `IQLConfig.disc_reward_sign` was silently unused** *(coupled to #1)*
- `IQLReplayBuffer.sample_step_batch` was hard-passing `r_disc = +intrinsic_reward(logit)` ignoring `cfg.disc_reward_sign`.
- Fix: replay now applies `negate_logit` (default) or `raw`, paired with Bug #1 so the agent is rewarded for being non-failure-like.

**Bug #3 — factory had no `"dipole_rl"` builder**
- Entry point `python -m robosuite.pipeline.train_dipole_rl` died with `KeyError: Unsupported algorithm type: dipole_rl`.
- Fix: added `@register_algorithm("dipole_rl")` decorator alongside the existing `"dipole"` builder in `robosuite/pipeline/factory.py`. Same `DipoleAgent.from_config` is used for both — the RL-specific pieces live in the trainer, not the agent.

**Bug #4 — `algorithm.dipole.lpb_detector.ckpt_path` defaulted to a non-existent path**
- The legacy `LPBV2GProvider` (kept around for the disc-display path and `g_mode=bce_frozen`) read `algorithm.dipole.lpb_detector.ckpt_path`, which defaulted to `checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth` in `train_dipole.yaml`. That path is not present on the verification machine even though `algorithm.discriminator.warm_start_ckpt` was set to a valid lpb_v2 BCE head.
- Fix: in `config/train_dipole_rl.yaml`, `algorithm.dipole.lpb_detector.ckpt_path` now interpolates from `algorithm.discriminator.warm_start_ckpt`, so the legacy provider and the SharedFrozenEncoder warm-start point at the same file by default. Override either via the wrapper script knobs (`LPB_CKPT` already sets `algorithm.discriminator.warm_start_ckpt`).

**Bug #5 — `q_learning.enabled` / `discriminator.online_train` flags were ignored**
- The integrated trainer constructed IQL and the online discriminator unconditionally. `train_step` then ran their updates regardless, emitting `iql/*` metrics even in the supposed-legacy regression run.
- Fix in `train_dipole_rl.py`:
  - Gate the IQL learner / replay construction on `cfg.algorithm.q_learning.enabled`.
  - Gate the disc / disc_replay construction on `cfg.algorithm.discriminator.online_train`.
  - Demo bootstrap routed via `disc_replay.bootstrap_from_demos` if disc is on; falls back to `agent.online_buffer.add` if only IQL is on; skipped entirely if both are off.
  - `g_mode=advantage` now hard-asserts both subsystems are on (otherwise AdvantageGProvider would receive `None` references and crash later).
  - IQL warmup loader prints `skipped (algorithm.q_learning.enabled=false)` instead of attempting to run.

**New artifact**
- `robosuite/pipeline/algorithms/discriminator/tests/test_encoder_sharing.py` (2 tests, cuda:1 required): asserts a single SharedFrozenEncoder ID reaches disc + AdvantageGProvider, and the head + Q/V + optim overhead on cuda:1 is bounded.

**Latency / FPS finding (filed; not in scope to fix)**
- Smoke fps ≈ 4.5 (500 env steps / 111 s, advantage mode). Below the `DIPOLE_RL.md §8` 10 fps threshold. Top cause: 3 redundant encoder forwards per learner tick (IQL replay s+s', disc replay s, AdvantageG s). The encoder is *physically* shared but *temporally* re-run on overlapping batches. A small `(start_index → context)` cache local to one `train_step` would collapse this. Out of scope for verification.

**Known shutdown issue (filed; not blocking)**
- At process shutdown the transition-chunk-writer flush deadline (`pipeline/utils/train_utils.py:551`, 120 s) is exceeded, raising `TimeoutError`. All metrics files are fully written *before* this fires, so the runs are still verifiable, but the non-zero exit code is misleading. Suggested fix: raise the deadline or drop pending chunks gracefully on shutdown.

**Notes for future subagents**
- The discriminator API now uses `prob_failure` (was `prob_expert`). If you are consuming `DiscriminatorOutput`, update field name.
- `disc.intrinsic_reward(...)` *still* returns the raw logit (higher = failure-like). The sign flip lives in `IQLReplayBuffer`, gated by `IQLConfig.disc_reward_sign`. Don't double-negate it.
- `DiscriminatorReplayBuffer.bootstrap_from_demos` no longer force-tags demos as expert — they fall into the non-failure pool naturally based on `is_intervention=False`. If you want a chunk to count as "failure" for the disc, ensure its window has at least one `is_intervention=True` transition.
- When `intervention.enabled=false`, the failure pool stays empty forever and `disc.update` is silently skipped (no `disc/*` metrics emitted that tick). This is correct — there is no signal to learn from — and tests should not require the `disc_loss` key in that configuration.

# Modification & Updates

### Subagent-8 · LPB v2 offline scorer for IQL r_disc + vis_qv — DONE (2026-05-23)

Fixes the bug captured in `pipeline/docs/IQL_DISCRIMINATOR_REWARD_DEBUG.md`:
IQL warmup (and `vis_qv.compute_qv_metrics`) used a frozen, partially-warm-started
`OnlineBCEDiscriminator` whose first `Linear` layer was reinitialized
(`in_dim=D_ctx` in the LPB v2 ckpt vs `D_ctx + H·D_a` in the chunk head). The
resulting `r_disc = -sigmoid(random_logit)` polluted Bellman targets and the
visualized failure plot.

**Files touched**
- `robosuite/pipeline/algorithms/discriminator/lpb_v2_scorer.py` — **new**. Public:
  - `LPBV2OfflineScorer(bce_ckpt_path, task_name, device, batch_size, meta_json_path=None)`:
    builds `BCEBenchmarkDiscriminator`; resolves operational tau in this order:
    (1) `<ckpt.parent>/meta.json["bce_youden_threshold"]` (loud about the source),
    (2) re-computed Youden via `_compute_youden_threshold_for_task` (success-calib
    + fail-bank from disk), (3) the ckpt's detector threshold as a clearly-
    flagged fallback.
  - `score_hdf5_demo(hdf5_path, demo_key, fps=20) -> np.ndarray[T]`: returns per-frame
    `failure_score = -g(z)` via `BCEBenchmarkDiscriminator.score_trajectory` — i.e.
    raw HDF5 image + state + TRUE action through the reference LPB v2 encoder, not
    the degraded `tile(proprio)` shared-encoder path.
  - `failure_score_to_margin_reward(scores)`: returns `tau - failure_score`.
    Convention: higher = more expert-like.
  - `annotate_transitions_with_lpb_scores(transitions, failure_scores, tau)`:
    pads/truncates to match `len(transitions)` and writes
    `lpb_failure_score / lpb_margin_reward / lpb_tau` onto each `Transition.info`.
  - All previous lpb-v2 helpers (`_load_lpb_bce_detector`, `_build_bce_discriminator_from_ckpt`,
    `_compute_youden_threshold_for_task`, `SafeRobosuiteBenchmarkTrajectory`,
    `_build_selected_trajectory`, `_load_failure_mask_and_segments`) live here;
    `vis_discriminator_util.py` now re-exports them so there is a **single
    scorer** for both reward and visualization.
- `robosuite/pipeline/algorithms/q_learning/utils/vis_discriminator_util.py` —
  thin viz layer importing from `lpb_v2_scorer`. `visualize_selected_trajectory_discriminator`
  now accepts an optional `scorer: LPBV2OfflineScorer | None` so callers can
  reuse a pre-built scorer (saves a model rebuild).
- `robosuite/pipeline/algorithms/q_learning/common.py` — `IQLConfig.disc_reward_source: Literal["online_bce", "precomputed_lpb"] = "online_bce"`
  (default keeps backward compat for any out-of-tree callers; the yaml default
  is `precomputed_lpb`).
- `robosuite/pipeline/algorithms/q_learning/replay.py` — `sample_step_batch` branches
  on `cfg.disc_reward_source`:
  - `precomputed_lpb`: reads `info["lpb_margin_reward"]` per step from each
    chunk (`_gather_precomputed_lpb`), aggregated with the same `γ`-discount as
    r_env. Missing-annotation transitions contribute 0 and trigger a one-shot
    warning.
  - `online_bce`: legacy `discriminator.intrinsic_reward(...) + disc_logit_to_intrinsic_reward`
    path, preserved as a backwards-compat / online-phase opt-in.
- `robosuite/pipeline/algorithms/q_learning/warmup.py` — builds `LPBV2OfflineScorer`
  once when `disc_reward_source=precomputed_lpb`, then iterates per (hdf5_path,
  demo_name) tuple inside `_load_split_into_buffer` (new helpers
  `_enumerate_split_demos` / `_group_transitions_by_episode`) and stamps
  `lpb_margin_reward` onto transitions before they enter the buffer. Skips the
  `OnlineBCEDiscriminator` construction entirely in this mode. Saves
  `encoder_meta["disc_reward_source"]` + `encoder_meta["lpb_scorer"]` (full
  summary dict) on the output ckpt so online runs can cross-check.
- `robosuite/pipeline/train_dipole_rl.py` — symmetric wiring: builds a single
  `LPBV2OfflineScorer` next to IQL/disc init; precomputes per-step LPB margins
  on the bulk-loaded offline demos (via `_enumerate_loaded_hdf5_demos` +
  `_group_transitions_by_episode`) before they enter the online buffer / disc
  replay. Online phase still uses the live `OnlineBCEDiscriminator` head — we
  only fixed the warmup/offline path per the agreed scope.
- `robosuite/pipeline/algorithms/q_learning/utils/vis_qv.py` — drops the
  `OnlineBCEDiscriminator` call inside `compute_qv_metrics`. The window's
  `disc_reward_horizon` is now `aggregate_chunk_reward(tau - failure_score[start:start+H], γ)`;
  `disc_logit` column reports `failure_score` at the chunk-start frame. The HUD
  video (`visualize_selected_trajectory_discriminator`) is reused as before but
  now shares the scorer instance — no double model build.
- `robosuite/pipeline/config/train_dipole_rl.yaml` — added
  `algorithm.q_learning.config.disc_reward_source: "precomputed_lpb"` as the
  new default, with an inline comment pointing at the debug doc.
- `robosuite/pipeline/algorithms/discriminator/tests/test_lpb_v2_scorer.py` —
  **new**. 9 CPU-only tests covering: `annotate_transitions_with_lpb_scores`
  shape / padding / truncation behavior, `_resolve_meta_json_tau` (happy path,
  missing file, malformed JSON), `IQLReplayBuffer._gather_precomputed_lpb`
  reads per-step margins from `info` (and warns when missing), and
  γ-aggregation matches the env-reward path.

**Hard contracts honored**
- `OnlineBCEDiscriminator` is **not modified**. Its replay buffer, update loop,
  and online-tick wiring in `train_dipole_rl.py` are untouched. Only the
  warmup path now bypasses it (and that path was the broken one).
- `LPBV2OfflineScorer` runs on the same `cuda:1` device as `SharedFrozenEncoder`
  (the LPB encoder load is independent of the shared one — the reference path
  needs the trained ViT, not the degraded `tile(proprio)` encoder).
- `vis_qv` CSV columns are stable: `disc_logit / disc_reward_chunk /
  disc_reward_horizon / total_reward_horizon` all still present, semantics
  updated (logit ≡ failure_score, chunk ≡ raw margin sum, horizon ≡ γ-aggregated).
- Tau resolution prefers `meta.json` — matches the user's note that the
  visualized / operational threshold is the Youden value in `meta.json`, not
  the detector threshold inside `bce_head.pth`.

**Verified**
- `py_compile` clean on all 7 touched + 1 new python files.
- `pytest robosuite/pipeline/algorithms/{q_learning,discriminator,dipole}/tests/`
  → **45 / 45 pass** (existing 36 + new 9), all CPU-only.

**Notes for downstream subagents**
- New `IQLConfig.disc_reward_source` knob: set to `precomputed_lpb` during
  warmup + offline; switch to `online_bce` (or just bump `disc_reward_coef`
  on top of `precomputed_lpb`) once the OnlineBCE chunk head is actually
  trained on intervention labels. The two are **not blended** today —
  pick one per phase.
- `LPBV2OfflineScorer` builds an independent `LPBV2Encoder` on the same
  device as `SharedFrozenEncoder`. They are NOT the same instance — the LPB
  reference path needs true actions; the shared encoder is the `tile(proprio)`
  degradation for Q/V context. If you need GPU pressure relief on cuda:1, the
  scorer can be moved to cuda:0 — feature dim matches.
- `encoder_meta["lpb_scorer"]` on the warmup ckpt is the structured tau audit
  trail (tau value + source string + ckpt path). Cross-check it against
  `cfg.algorithm.discriminator.warm_start_ckpt` on the online run.


### Subagent-9 · DEBUG_and_ERRORs.md fixes — DONE (2026-05-24)

Addresses items 1–4 of `pipeline/docs/DEBUG_and_ERRORs.md`. Item 5
(tensorboard migration) deferred to its own PR.

**Files touched**
- `robosuite/pipeline/algorithms/discriminator/online_bce.py` —
  `intrinsic_reward(...)` now returns
  `-sigmoid(logit - threshold) ∈ (-1, 0)` directly. Sign convention
  baked in: failure-like → ≈ −1 (penalty), expert-like → ≈ 0 (no
  penalty), aligned with env reward (−1 per step / 0 success). The
  EMA-tracked decision `threshold` is unchanged (median of
  non-failure-pool logits, `threshold_ema=0.99` default).
- `robosuite/pipeline/algorithms/q_learning/common.py` — dropped
  `IQLConfig.disc_reward_source` and `IQLConfig.disc_reward_sign`.
  `disc_reward_coef` is the only remaining knob; reward composition is
  `r_total = r_env + disc_reward_coef · r_disc` with `r_disc` already
  in (-1, 0).
- `robosuite/pipeline/algorithms/q_learning/replay.py` —
  `sample_step_batch` collapsed to a single online-disc path. Per-chunk
  `r_disc` is broadcast across the H-step chunk **without the legacy
  `/H` scale** so each step contributes the full intrinsic reward, then
  `aggregate_chunk_reward` applies γ-discount the same way as r_env.
  `_gather_precomputed_lpb` removed. `_stack_views_uint8` no longer
  calls `_center_crop_resize` — the SharedFrozenEncoder owns the LPB
  v2 preprocessing chain (F.interpolate → LinearNormalizer →
  CenterCrop) end-to-end (DEBUG §4).
- `robosuite/pipeline/algorithms/q_learning/data_util.py` — removed
  `disc_logit_to_intrinsic_reward` (the sign flip now lives inside the
  discriminator).
- `robosuite/pipeline/algorithms/q_learning/warmup.py` — removed
  `LPBV2OfflineScorer` construction, `disc_reward_source` branch,
  `annotate_transitions_with_lpb_scores` call, and the related
  `_enumerate_split_demos / _group_transitions_by_episode` helpers.
  Warmup always builds a frozen `OnlineBCEDiscriminator` (warm-started
  from the LPB v2 BCE head, `.update()` never called). Saved
  `encoder_meta` no longer carries `lpb_scorer` / `disc_reward_source`.
- `robosuite/pipeline/algorithms/discriminator/replay.py` — same
  `_stack_views_uint8` signature update.
- `robosuite/pipeline/algorithms/discriminator/encoder.py` — docstring
  on `encode(...)` documents that callers must NOT pre-crop / pre-
  resize.
- `robosuite/pipeline/algorithms/discriminator/base.py` — docstrings
  updated to reflect the new reward convention.
- `robosuite/pipeline/algorithms/dipole/agent.py` —
  `build_checkpoint_payload(...)` now includes `iql_state` and
  `discriminator_state` when the corresponding components are attached
  via `attach_iql_learner / attach_discriminator`. Debug-only — absence
  is fine for legacy DIPOLE runs.
- `robosuite/pipeline/train_dipole_rl.py` — removed
  `_enumerate_loaded_hdf5_demos`, `_group_transitions_by_episode`,
  `LPBV2OfflineScorer` build, the offline-demo precompute block, and
  the `list_hdf5_demo_names` import. Online path is unchanged
  otherwise.
- `robosuite/pipeline/algorithms/q_learning/utils/vis_qv.py` — dropped
  the metadata field referencing the removed `disc_reward_sign`.
- `robosuite/pipeline/config/train_dipole_rl.yaml` — removed
  `disc_reward_source` and `disc_reward_sign`; reward-side comment
  now points at the unified path.
- Tests: trimmed `test_iql.py::test_disc_logit_to_intrinsic_reward_sigmoid_range`
  and two `test_lpb_v2_scorer.py` cases exercising the removed
  `_gather_precomputed_lpb` / `IQLConfig(disc_reward_source=...)`.
  Added `test_online_bce.py::test_intrinsic_reward_formula_matches_sigmoid_threshold`
  asserting `intrinsic_reward == -sigmoid(logit - threshold)` exactly,
  and updated `test_intrinsic_reward_sign` to check the new in-range
  reward (`r_failure.mean < r_non_failure.mean`).

**What survives**
- `LPBV2OfflineScorer` (`discriminator/lpb_v2_scorer.py`) is kept —
  the `vis_qv` visualization path still scores HDF5 demos with the
  reference LPB v2 BCE detector for the failure-plot overlay. It no
  longer participates in the IQL reward path.
- Legacy `LPBV2GProvider` is unchanged; still used for
  `g_mode=bce_frozen` and the disc-display overlay during rollout.

**Hard contracts (still honored)**
- One daemon learner thread; flow → IQL → disc per tick.
- `DipoleFlowPolicy.update()` signature unchanged.
- Encoder ownership stays in the trainer; injected into IQL / disc /
  AdvantageG; never trained.

**Verified**
- `python -m py_compile` clean on every touched file.
- `pytest robosuite/pipeline/algorithms/{q_learning,discriminator,dipole}/tests/ -v`
  → **43 / 43 pass** (was 45; trimmed 3 tests for removed APIs, added
  1 micro-test for the new formula).
- `from robosuite.pipeline import train_dipole_rl` imports cleanly;
  same for `q_learning.warmup`.

**Headless smoke target (for the next runner)**
```bash
INTERACTIVE=false VIEWER_ENABLED=false INTERVENTION_ENABLED=false \
  INIT_CHECKPOINT=<path> bash robosuite/pipeline/scripts/train_dipole_rl.sh \
  runtime.max_steps=500 algorithm.trainer.pretrain_steps=20 \
  algorithm.q_learning.warmup_value_steps=20 \
  algorithm.q_learning.warmup_full_steps=10 \
  logging.checkpoint_interval=200
```
Expected per-tick metric keys: `flow_loss / iql/q_loss / iql/v_loss /
disc/disc_loss / disc/threshold / G_mean / advantage_mean /
disc_reward_mean`. `disc_reward_mean` must lie in (-1, 0). A ckpt at
step 200 must contain top-level keys `iql_state` and
`discriminator_state` alongside the existing policy fields.

**Notes for downstream subagents**
- `OnlineBCEDiscriminator.intrinsic_reward` no longer returns the raw
  logit — it returns the reward directly. If you need the logit, use
  `disc.score(...).logit` (unchanged).
- IQL reward is now entirely owned by the discriminator's sign
  convention. Do NOT re-negate at the replay boundary; the old
  `disc_reward_sign` knob is gone.
- Encoder image preprocessing is owned by `SharedFrozenEncoder.encode`.
  Don't pre-resize / pre-crop in any new caller.
- `iql_state` / `discriminator_state` in the policy ckpt are
  debug-only snapshots. There is no resume hook for them yet; load
  them manually with `IQLLearner.load_state_dict /
  OnlineBCEDiscriminator.load_state_dict` if you want to inspect a
  past Q/V or BCE head.
- `DEBUG_and_ERRORs.md` items 1–4 are now closed; item 5 (tensorboard
  migration) remains open.


### Subagent-10 · Single-frame `OnlineBCEDiscriminator` (lpb_v2-aligned) — DONE (2026-05-24)

Aligns the online BCE discriminator with the reference single-frame head
in `robosuite/discriminator/lpb_v2/detectors/bce.py`. The head's input
dim drops from `D_ctx + H·D_a` to `D_ctx`, so warm-start from the lpb v2
ckpt now loads **every** layer including the first Linear — previously
that layer was silently re-initialized because the chunk-wise head's
shape didn't match. The IQL replay encodes every frame of the chunk
with the **real per-step action** and γ-aggregates the per-frame disc
rewards through the same `aggregate_chunk_reward` path that handles
`r_env ∈ {-1, 0}`.

**Files touched**
- `robosuite/pipeline/algorithms/discriminator/encoder.py` — `SharedFrozenEncoder.encode(...)`
  gains an optional `action_real: (B, D_a)` kwarg. When supplied, the
  encoder forwards `(image, proprio, action_real)` to `LPBV2Encoder.encode_batch`
  matching the lpb v2 training distribution. When omitted, the legacy
  `tile(proprio)` degraded path is preserved for backward compatibility
  (legacy `LPBV2GProvider` calls and the disc-display overlay). New
  helper `_adapt_action_to_input_dim` mirrors the shape contract of
  `_tile_proprio_as_action`.
- `robosuite/pipeline/algorithms/discriminator/base.py` —
  `DiscriminatorBatch` drops `action_chunk`; `DiscriminatorBase.score /
  intrinsic_reward` signatures collapse to `(context,) -> …`.
- `robosuite/pipeline/algorithms/discriminator/online_bce.py` —
  `OnlineBCEDiscriminator.__init__` drops the `action_horizon` kwarg;
  `in_dim = D_ctx`. `_featurize`, `score`, `intrinsic_reward`, `update`
  all consume per-frame `context (B, D_ctx)` only. `state_dict` no
  longer serializes `action_horizon`; loading older ckpts that carry
  it is a silent no-op for back-compat. Reward formula unchanged:
  `r = -sigmoid(logit - threshold) ∈ (-1, 0)`.
- `robosuite/pipeline/algorithms/discriminator/bce_head.py` —
  `warm_start_from_lpb_bce_ckpt` adds a **hard assertion** that
  `net.0.weight` is in the compatible-shape set. With the single-frame
  redesign this is now true by construction; the assert prevents silent
  regression if `in_dim` ever drifts away from `D_ctx`.
- `robosuite/pipeline/algorithms/discriminator/replay.py` —
  `DiscriminatorReplayBuffer` is now per-frame:
  - bucket keys are storage indices (not chunk-start indices),
  - label = `transition.is_intervention ∈ {0, 1}`,
  - `sample(...)` reads a single transition per sample and calls
    `encoder.encode(..., action_real=transition.action)` to land in the
    lpb v2 BCE head's training domain,
  - the constructor drops the `action_horizon` kwarg.
- `robosuite/pipeline/algorithms/q_learning/replay.py` —
  `IQLReplayBuffer.sample_step_batch` now:
  - collects per-step `(image, proprio, action_real)` for every step of
    each H-chunk,
  - encodes the flattened `(B·H, …)` batch in one `encoder.encode(...)`
    call (returns `(B·H, D_ctx)`, reshapes to `(B, H, D_ctx)`),
  - calls `discriminator.intrinsic_reward(context=(B·H, D_ctx))` once
    and reshapes back to `(B, H)`,
  - feeds the per-step `r_env + λ·r_disc` through
    `aggregate_chunk_reward(γ)` exactly like r_env alone.
  Cost: H+1 encoder forwards per chunk (vs 2 previously). `IQLStepBatch.context`
  is still `(B, D_ctx)` (chunk-start latent), so the Q-network input
  shape is unchanged.
- `robosuite/pipeline/algorithms/dipole/advantage_g_provider.py` —
  `compute_g_for_batch` encodes context with `action_real =
  batch.action_sequences_raw[:, 0, :]` (chunk-start action, consistent
  with IQL's chunk-start context). `disc.score` call drops
  `action_chunk=`.
- `robosuite/pipeline/train_dipole_rl.py` + `robosuite/pipeline/algorithms/q_learning/warmup.py` —
  `OnlineBCEDiscriminator(...)` construction loses the `action_horizon`
  kwarg; `DiscriminatorReplayBuffer(...)` loses it too.
- Tests: 9 tests in `test_online_bce.py` rewritten for the single-frame
  API; `test_warm_start_loads_first_linear` now asserts `net.0.weight`
  equals the ckpt tensor bit-for-bit. `test_shared_encoder.py` gains
  `test_action_real_affects_latent`. `test_advantage_g_provider.py` and
  `test_encoder_sharing.py` updated to the new `score(context=…)` /
  no-`action_horizon` signatures.

**Hard contracts honored**
- One daemon learner thread; flow → IQL → disc per tick.
- `DipoleFlowPolicy.update()` signature unchanged.
- Encoder ownership stays in the trainer; never trained.
- Legacy `LPBV2GProvider` (used for `g_mode=bce_frozen` and the
  disc-display overlay) is untouched; the `action_real=None` path keeps
  it bit-compatible.

**Verified**
- `python -m py_compile` clean on every touched file.
- `pytest robosuite/pipeline/algorithms/{q_learning,discriminator,dipole}/tests/`
  → **49 / 49 pass** (was 43; +1 single-frame warm-start regression,
  +1 `featurize` context-dim guard, +1 legacy `action_horizon` ignore,
  +1 `action_real` affects latent, +2 net new from existing modules).

**Notes for future subagents**
- `OnlineBCEDiscriminator.score / intrinsic_reward` are single-frame
  now: `score(context=(B, D_ctx))`, no `action_chunk` kwarg. The
  encoder owns the action concat via `action_real`.
- `DiscriminatorBatch` no longer carries `action_chunk`; if you're
  constructing one, drop that field.
- For per-step disc rewards, just call `discriminator.intrinsic_reward(
  context=flat_ctx)` on a flattened `(B·H, D_ctx)` tensor and reshape;
  this is the shape the IQL replay feeds in.
- The `(start_idx → context)` caching mentioned in §8 is still open —
  the per-step encoding makes encoder fps the dominant cost on the
  learner GPU. Cache key candidate: `(storage_index, encoder_id)`.

