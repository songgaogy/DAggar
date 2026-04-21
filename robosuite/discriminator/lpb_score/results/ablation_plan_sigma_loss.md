# Retrain Ablation Plan — σ × Loss-Branch (post-V6)

## 0. Scope and non-scope

**In scope** (this round):
- Ablate **noise scale σ** (`model.noise_scale`).
- Ablate **loss branch composition** (`state_loss_weight`, `dynamics_loss_weight`).
- All runs use **EMA on + dropout=0.1** (cheap stability wins locked in).

**Explicitly out of scope**:
- Frame-level failure labels — **do not use.** Keep per-trajectory
  `traj_type = 1 if data_type == "fail_rollout" else 0` (current
  `dataset.py:491`). The user has locked this.
- Architecture changes (L1/L2 CFG separation, cross-attention, multi-σ
  conditioning).
- α branch revival in scoring (still β-only at inference; no need to save α
  machinery during training per scoring side — but both branches still
  train identically regardless of α usage).

## 1. Experimental design

Two independent axes. Full cross is 4 σ × 3 branch = 12 runs; we use a
**two-phase** design to cut this to 6.

### Phase A — σ sweep (loss held at both=1)

Purpose: find σ\* where β-only detector is best. Also records α's potential
revival as a diagnostic (α still evaluated at benchmark time even if we
won't deploy it).

### Phase B — branch sweep (σ held at σ\*)

Purpose: quantify how much each factor (state, dynamics) contributes to
β-based discrimination.

### Common settings (all runs)

```yaml
training.use_ema: true
training.ema_decay: 0.999
training.val_use_ema: true
training.save_ema_in_checkpoint: true
model.dropout: 0.1

# Everything else identical to V6 config/train.yaml:
training.batch_size: 256
training.epochs: 50
training.lr: 2e-4
training.weight_decay: 1e-4
training.positive_ratio: 0.5
training.grad_clip_norm: 1.0
model.embed_dim: 512
model.num_layers: 4
model.ffn_dim: 2048
model.std_clamp_min: 0.05
data.transition_horizon: 1
# Same 6 tasks, same split counts (360/180 train per task).
```

Each run gets a unique `save_dir` tag:
`checkpoints/multitask_6/lpb_score/abl_sig{σ}_lw{state}_{dyn}_ema/`.

---

## 2. Experiments table

### Phase A — σ sweep  (state_loss=1, dynamics_loss=1, EMA on, dropout=0.1)

| run_id | noise_scale σ | state_lw | dyn_lw | EMA | dropout | notes |
|---|:-:|:-:|:-:|:-:|:-:|---|
| A1 | **0.04** | 1.0 | 1.0 | on | 0.1 | σ < V6; tighter manifold, risk of under-denoising signal |
| A2 | **0.08** | 1.0 | 1.0 | on | 0.1 | **V6 baseline + EMA/dropout** (isolates EMA/dropout effect) |
| A3 | **0.12** | 1.0 | 1.0 | on | 0.1 | σ > V6; looser manifold, may revive α branch |
| A4 | **0.20** | 1.0 | 1.0 | on | 0.1 | σ » V6; expected to clearly widen E⁺ distribution |

Expected diagnostic from A1–A4:
- `[collapse]` epoch log: `state_diff_l2` / `dynamics_diff_l2` should be
  monotone-ish in σ. Too small σ → collapse; too large σ → flat LLR.
- α-branch benchmark (rerun sweep with `ALPHA_STATE=ALPHA_DYNAMICS=1`):
  track whether α regains any signal.
- β-only benchmark (Config B from V6 summary: `max, window=-1, β=1`): pick
  σ\* = argmax frame_AUROC (tie-break traj_AUROC).

### Phase B — branch sweep  (σ = σ\* from Phase A, EMA on, dropout=0.1)

| run_id | noise_scale σ | state_lw | dyn_lw | EMA | dropout | notes |
|---|:-:|:-:|:-:|:-:|:-:|---|
| B1 | σ\* | **1.0** | **0.0** | on | 0.1 | state-only: `p(z_t|c)` factor alone |
| B2 | σ\* | **0.0** | **1.0** | on | 0.1 | dynamics-only: `p(z_{t+H}|z_t, a, c)` factor alone |
| B3 | σ\* | 1.0 | 1.0 | on | 0.1 | **same as A's σ\* run — skip if already done** |

If σ\* = 0.08 (so B3 duplicates A2), total new runs = **4 (Phase A) + 2 (Phase B) = 6**.
If σ\* ≠ 0.08, need to train B3 fresh too: total 4 + 3 = 7 runs.

**Evaluation for Phase B runs**: the inference scorer expects both heads
to exist; B1/B2 still load the full model — we just have one head with
near-random outputs. At benchmark time we set the corresponding β to 0:

- For B1 (state-only trained): evaluate with `β_state=1, β_dyn=0`.
- For B2 (dynamics-only trained): evaluate with `β_state=0, β_dyn=1`.
- For B3 (both): evaluate with `β_state=1, β_dyn=1`.

This isolates **per-factor discrimination power**.

---

## 3. Success criteria

- **σ\* from Phase A**: pick the σ maximizing frame_AUROC under Config B.
  Accept as σ\* if it beats A2 (V6+EMA/dropout) by at least +0.01
  frame_AUROC. Otherwise σ\* = 0.08.
- **Branch necessity from Phase B**: if B1 or B2 alone comes within 0.01
  traj_AUROC and 0.02 frame_AUROC of B3, the other branch is
  under-contributing — architectural simplification candidate. If both B1
  and B2 drop ≥ 0.05 vs B3, we have confirmed factor complementarity.
- **EMA/dropout check from A2**: A2 vs V6 (0.9493 traj / 0.9228 frame
  under Config B) tells us how much EMA+dropout alone buys. Expected:
  +0.002–0.005 traj, similar or slightly higher frame.

## 4. Benchmark protocol for each run

Use the existing sweep harness. For each retrained checkpoint:

1. Point `DSM_CKPT` in `scripts/run_lpb_score_benchmark.sh` at the new checkpoint.
2. Run the **V6 Config B default** (`LAMBDA_MODE=max, LAMBDA_WINDOW_SIZE=-1`,
   `β=1`, `α=0`) as the canonical baseline.
3. Also run **Config A** (`mean, 50, β=1`) for traj_AUROC reference.
4. For Phase A diagnostics, also run once with `α=1` to check if σ revived α.
5. Log the `[collapse]` diagnostic means per epoch from training; flag any
   run where `state_diff_l2 < 1e-4` or `dynamics_diff_l2 < 1e-4` for ≥3
   consecutive epochs (Trainer already does this).

## 5. Commands (templates)

### Train one variant

```bash
# Example: A3 (σ=0.12, both losses, EMA, dropout=0.1)
/home/dodo/miniconda3/envs/daggar/bin/python -m robosuite.discriminator.lpb_score.train \
  model.noise_scale=0.12 \
  model.state_loss_weight=1.0 \
  model.dynamics_loss_weight=1.0 \
  model.dropout=0.1 \
  training.use_ema=true \
  training.ema_decay=0.999 \
  save_dir=./checkpoints/multitask_6/lpb_score/abl_sig0.12_lw1.0_1.0_ema \
  save_name=lpb_score_dsm.pt
```

Or equivalently via a thin wrapper shell — recommended to add
`scripts/train_lpb_score_dsm_abl.sh` taking `SIGMA`, `STATE_LW`, `DYN_LW`
env vars and dispatching to the Hydra CLI. Follow the same style as
`train_lpb_score_dsm.sh`.

### Benchmark one variant

```bash
DSM_CKPT=checkpoints/multitask_6/lpb_score/abl_sig0.12_lw1.0_1.0_ema/lpb_score_dsm.pt \
LAMBDA_MODE=max LAMBDA_WINDOW_SIZE=-1 \
ALPHA_STATE=0 ALPHA_DYNAMICS=0 BETA_STATE=1 BETA_DYNAMICS=1 \
bash robosuite/discriminator/lpb_score/scripts/run_lpb_score_benchmark.sh
```

---

## 6. Results template (fill in as runs complete)

Report each run's benchmark under **Config B** (default) and **Config A** (traj-max):

| run_id | σ | state_lw | dyn_lw | trA (B) | frA (B) | evP (B) | trA (A) | frA (A) | collapse? |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| V6 (ref) | 0.08 | 1.0 | 1.0 | 0.9493 | 0.9228 | 0.984 | 0.9728 | 0.8422 | no |
| A1 | 0.04 | 1.0 | 1.0 |  |  |  |  |  |  |
| A2 | 0.08 | 1.0 | 1.0 |  |  |  |  |  |  |
| A3 | 0.12 | 1.0 | 1.0 |  |  |  |  |  |  |
| A4 | 0.20 | 1.0 | 1.0 |  |  |  |  |  |  |
| B1 | σ\* | 1.0 | 0.0 |  |  |  |  |  |  |
| B2 | σ\* | 0.0 | 1.0 |  |  |  |  |  |  |

Commit this filled table back to `results/` when Phase A/B complete.
