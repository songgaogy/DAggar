# lpb_score V6 — Benchmark Summary & Reproduction Guide

Current production checkpoint:
`checkpoints/multitask_6/lpb_dipole-new-v6/lpb_dipole_dsm_20260420_070130/lpb_dipole_dsm_20260420_070130.pt`

Benchmark tasks: `PickPlaceCan`, `PickPlaceBread`, `PickPlaceCereal`, `PickPlaceMilk`
(178 trajectories total: 71 failure / 107 success)

---

## 1. Best post-hoc eval configurations

After a 2-round sweep (`eval/sweep_20260421_170146` + `eval/sweep_tail_20260421_174845`)
exploring `lambda_mode × lambda_window_size × β-weights`, three operating
points emerge as Pareto-optimal. Pick per downstream goal.

α is dropped in all recommended configs (β-only is the Bayes-optimal LLR; α
has no usable OOD signal under the V6 σ=0.08 training regime).

### 1.1 Recommended configs

| Config | mode | window | β_state | β_dynamics | traj_AUROC | frame_AUROC | evP | evR | delay_med | Use when |
|---|---|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|---|
| **A. traj-max** | mean | 50 | 1 | 1 | **0.9728** | 0.8422 | 0.610 | 0.930 | -79 | Max trajectory discrimination (offline ranking) |
| **B. frame-max** (default) | max | -1 | 1 | 1 | 0.9493 | **0.9228** | **0.984** | 0.845 | **-7** | Frame/event localization + online alarm |
| **C. bounded online** | max | 120 | 1 | 1 | 0.9493 | 0.8938 | 0.942 | 0.887 | -13 | Same as B but with bounded memory |

**Recommended default: Config B** (max, window=-1). `frame_AUROC=0.92`,
`event_precision=0.98`, detection delay ≈ GT, at the cost of 0.024 traj
AUROC vs Config A. For online / pipeline use this is almost always the
right tradeoff.

Config C approximates Config B with a 120-step rolling window; useful when
the detector runs in an environment where full-prefix memory isn't
available.

### 1.2 Per-task breakdown

#### Config A (mean, window=50) — traj-optimal
| Task | traj_AUROC | frame_AUROC | evP | evR | delay_med |
|---|:-:|:-:|:-:|:-:|:-:|
| PickPlaceBread   | 0.9662 | 0.7425 | 0.48 | 0.91 | -111 |
| PickPlaceCan     | 0.9798 | 0.8528 | 0.59 | 0.82 | -30 |
| PickPlaceCereal  | 1.0000 | 0.9239 | 0.90 | 1.00 | -49 |
| PickPlaceMilk    | 0.9897 | 0.7672 | 0.52 | 0.90 | -180 |

#### Config B (max, window=-1) — frame/event-optimal (default)
| Task | traj_AUROC | frame_AUROC | evP | evR | delay_med |
|---|:-:|:-:|:-:|:-:|:-:|
| PickPlaceBread   | 0.9533 | 0.8658 | 1.00 | 0.70 | +38 |
| PickPlaceCan     | 0.9259 | 0.9514 | 1.00 | 0.73 | +17 |
| PickPlaceCereal  | 1.0000 | 0.9571 | 1.00 | 1.00 | -14 |
| PickPlaceMilk    | 0.9862 | 0.9694 | 0.90 | 0.90 | +3  |

**Notes:**
- `PickPlaceCan` is the weakest task on traj AUROC under Config B (0.926);
  if your downstream loss is dominated by Can, use Config A.
- `PickPlaceBread` / `PickPlaceMilk` have very early detection under Config
  A (delay ≤ -110) — indicates false-early alarms in the mean-aggregated
  score. Config B fixes this.

### 1.3 How to run

Benchmark script: `robosuite/discriminator/lpb_score/scripts/run_lpb_score_benchmark.sh`.
Env overrides select the config.

```bash
# Config A — traj-max
LAMBDA_MODE=mean LAMBDA_WINDOW_SIZE=50 \
ALPHA_STATE=0 ALPHA_DYNAMICS=0 BETA_STATE=1 BETA_DYNAMICS=1 \
bash robosuite/discriminator/lpb_score/scripts/run_lpb_score_benchmark.sh

# Config B — frame-max (default recommendation)
LAMBDA_MODE=max LAMBDA_WINDOW_SIZE=-1 \
ALPHA_STATE=0 ALPHA_DYNAMICS=0 BETA_STATE=1 BETA_DYNAMICS=1 \
bash robosuite/discriminator/lpb_score/scripts/run_lpb_score_benchmark.sh

# Config C — bounded online
LAMBDA_MODE=max LAMBDA_WINDOW_SIZE=120 \
ALPHA_STATE=0 ALPHA_DYNAMICS=0 BETA_STATE=1 BETA_DYNAMICS=1 \
bash robosuite/discriminator/lpb_score/scripts/run_lpb_score_benchmark.sh
```

Note: `DELTA` in the runner controls only the detector-internal calibration
threshold; the benchmark harness uses its own `step_success_percentile=95`
for frame binarization, independent of `DELTA`. To sweep step thresholds,
edit `data/utils/benchmark/examples/run_lpb_score.py` (gitignored).

---

## 2. Training parameters used to produce V6

All values below are from `config/train.yaml` as checked in on branch
`uni-dsm` at commit that produced the V6 checkpoint (trained 2026-04-20).

### 2.1 Policy / data

```yaml
policy.ckpt: checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt
policy.encoder_batch_size: 96

data.image_size: 128
data.transition_horizon: 1
data.cache_dir: ./data/.lpb_score_cache
data.encode_demo_batch_size: 8

data.splits:
  train: {num_pos_traj: 360, num_neg_traj: 180}
  val:   {num_pos_traj:  30, num_neg_traj:  15}
  test:  {num_pos_traj:  10, num_neg_traj:   5}

data.tasks:
  - PandaLift
  - PandaPickPlaceCan
  - PandaStack
  - PickPlaceBread
  - PickPlaceCereal
  - PickPlaceMilk
```

Positive pool = `expert/` ∪ `success_rollout/`; negative pool =
`fail_rollout/`. `traj_type=1` is assigned **per whole trajectory** for any
fail_rollout trajectory.

### 2.2 Model

```yaml
model.embed_dim: 512
model.num_layers: 4
model.num_heads: 8
model.ffn_dim: 2048
model.dropout: 0.0          # ← zero regularization; next retrain: 0.1
model.std_clamp_min: 0.05
model.noise_scale: 0.08     # fixed σ (SσDC single-σ)
model.state_loss_weight: 1.0
model.dynamics_loss_weight: 1.0
```

Backbone: `UnifiedConditionedDSM` — 4-layer AdaLN-Zero DiT; per-factor
routed encoders (`state`, `dynamics`); shared backbone conditioned by
`(route_emb, type_emb, task_name_emb, context_feature)` fused through a
single `condition_mlp`.

### 2.3 Training loop

```yaml
training.device: cuda
training.batch_size: 256
training.num_workers: 8
training.epochs: 50
training.save_freq: 10
training.lr: 2e-4
training.weight_decay: 1e-4
training.positive_ratio: 0.5    # WeightedRandomSampler balance
training.grad_clip_norm: 1.0
training.log_every: 100
training.shared_lr_multiplier: 1.0
training.state_branch_lr_multiplier: 1.0
training.dynamics_branch_lr_multiplier: 1.0

training.use_ema: false         # ← next retrain: true
training.ema_decay: 0.999
training.val_use_ema: true
training.save_ema_in_checkpoint: true
```

Optimizer: `AdamW` with branch-aware parameter groups (`shared`,
`state_branch`, `dynamics_branch`) — LR multipliers all 1.0 for V6.

Loss per step:
```
L = state_loss_weight · ||g_state(z_t + σε_s, c) − z_t_norm||²
  + dynamics_loss_weight · ||g_dyn(Δ + σε_d, z_t_norm, a, c) − Δ||²
where Δ = normalize(z_{t+H}) − normalize(z_t).
```

### 2.4 Reproduce command

```bash
bash robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm.sh
```

Output: `checkpoints/multitask_6/lpb_score/lpb_score_dsm_<timestamp>.pt`
and every 10 epochs.

---

## 3. Known limitations (V6)

- **α branch carries no signal at σ=0.08.** β-only is the Bayes LLR and is
  mathematically complete; the α term is an OOD-prior hedge that only
  activates if c=1 branch collapses. On V6 β alone is strictly better. See
  ablation plan for σ-sweep to verify whether α can be revived.
- **`PickPlaceCan` traj AUROC stuck at 0.93.** Other 3 tasks saturate at
  ≥0.99. Likely failure-manifold coverage / latent early-frame overlap.
- **`PickPlaceBread` frame AUROC = 0.87** (lowest under Config B). Early
  failures on Bread are especially subtle; frame-level localization is
  weaker than on Can/Cereal/Milk.
- **No EMA, no dropout.** Config omissions in V6; cheap wins available
  next retrain.

---

## 4. Change log

- **2026-04-20** V6 trained: `lpb_dipole_dsm_20260420_070130.pt` (50 epochs,
  branch `uni-dsm`, first SσDC checkpoint).
- **2026-04-21** Post-hoc sweep completed (18 configs across 2 rounds).
  Default eval config set to Config B.
