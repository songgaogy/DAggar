# DIPOLE-RL — DIPOLE + IQL Q-chunking + Online BCE Discriminator

This document is the design spec for extending DIPOLE (currently CFG-style
polarity flow with a **frozen** LPB v2 BCE detector) into a fully online
human-in-the-loop RL pipeline.

The DIPOLE algorithm itself (steps 1–4 of the project plan) is already in
place — see `DIPOLE.md`. This document specifies steps **(5) offline Q/V
warmup** and **(6) online co-training of flow policy + Q/V + discriminator**.

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

The DIPOLE flow policy code (`models/flow.py`) is unchanged. The only
hook is `agent.attach_g_provider(...)`, which now receives
`AdvantageGProvider` instead of `LPBV2GProvider`.

---

## 2. Architecture

```
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

1. Sample `step_batch` (chunk + reward + next_obs + done) and `actor_batch`
   (raw chunk + context) from the shared replay store.
2. `no_grad`: re-evaluate `r_disc` on `step_batch` with the **current** disc
   to avoid stale rewards; overwrite `step_batch.rewards`.
3. IQL Q loss: Bellman MSE for both critics.
4. IQL V loss: expectile(`min(Q1, Q2) − V`).
5. `no_grad`: compute `A` on `actor_batch` and `disc_logit` on its
   `(context, action_chunk_raw)`; normalize each; combine into G.
6. DIPOLE flow update with G (replaces existing
   `LPBV2GProvider.compute_g_for_batch(...)` path).
7. Sample balanced expert/policy batch from disc replay; one BCE step.
8. Polyak target V update.
9. Emit metrics; return to async queue.

Main thread: rollout → env step → intervention routing →
`trainer.record_transition` (which fans out into DIPOLE buffer, IQL
buffer, and disc buffer) → optional `sync_inference_policy`.

### Reward composition

```
r_total(s, a) = r_env(s, a) + λ_disc · r_disc(s, a)
              = (−1 step / 0 success) + λ_disc · (−disc_logit_normalized)
```

`r_total` is what IQL bootstraps off. The discriminator term is **always**
computed under no_grad and uses the current discriminator weights (re-eval
on sampled batch, not stored stale).

### G composition

```
A(s, a)        = min(Q1(s, a), Q2(s, a)) − V(s)          [IQL learner, no_grad]
disc_logit     = OnlineBCEDiscriminator.score(s, a).logit [no_grad]
A_norm         = batch_zscore(A)
disc_norm      = batch_zscore(disc_logit)
G              = α · A_norm + β · (−disc_norm)
```

DIPOLE then maps G to polarity weights `w_pos = sigmoid(β_dipole · G + k)`
exactly as today.

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
- **Encoder must stay frozen**: no `encoder.train()`, no gradients through
  encoder. Any forward must be inside `torch.no_grad()`.
- **Q-chunk input shape is fixed**: `D_ctx + H·D_a` for Q; `D_ctx` for V.
  No per-step Q factorization.
- **Concurrency is fixed**: one daemon learner thread on cuda:1; no
  multiprocessing, no DDP.
- **`DipoleFlowPolicy.update()` signature must not change**: G enters
  exclusively via `compute_g_for_batch(batch) → (B,)`.
- **Markdown lives only under `pipeline/docs/`**; no new READMEs anywhere
  else.

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

`DipoleConfig` gets a new field `g_mode: Literal["bce_frozen", "advantage"]
= "bce_frozen"`. Default preserves backward compatibility for any existing
DIPOLE run.

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
2. Encoder identity check: `id(trainer.encoder) is
   id(iql.replay._encoder_ref) is id(disc._encoder_ref)`.
3. IQL warmup smoke: 200 steps, no NaN, V loss monotonic-ish down.
4. Headless integrated smoke: 500 env steps, metrics dict contains
   `flow_loss / q_loss / v_loss / disc_loss / G_mean / advantage_mean /
   disc_reward_mean`, no NaN/Inf.
5. `nvidia-smi`: cuda:0 has inference process; cuda:1 has training process;
   each < 20 GB.
6. Regression: `algorithm.dipole.g_mode=bce_frozen` reproduces existing
   DIPOLE behavior bit-for-bit on a fixed seed (or as close as fp64-vs-fp16
   noise allows).

---

## 8. Known risks (carry forward)

- **Q-chunking with off-policy actions** is standard IQL; we evaluate
  historical-policy chunks. Document in `01_q_learning_iql`.
- **Single learner thread throughput**: if fps drops below 10, fall back to
  `disc.cfg.update_every_n_steps > 1`.
- **Discriminator cold start**: gate the advantage path via
  `runtime.bootstrap_g_with_frozen_bce` until warmup completes.
- **Intervention scarcity**: balanced sampler uses replacement on the
  expert pool.

See `DIPOLE.md` for the original CFG-style polarity weighting details
(unchanged) and `README.md` for the entry-point commands.
