# D⁴-Disc 0424 debug: "collapse to LPB" baseline (phase-A only, ω=0)

> Context: user ran a *phase-A-only* D⁴-Disc training (`SKIP_BOOTSTRAP=1`) and
> evaluated with `ω=0` as a LPB-parity baseline. Both runs on this date had
> essentially chance-level trajectory AUROC even though per-frame signal looked
> fine. Two distinct bugs were found and fixed; this doc records both.

## 0. Summary of final state

| run | eval config | pooled traj AUROC | notes |
|---|---|---|---|
| `run_20260423_232821` | abs + max (old) | 0.503 | **bug 1**: encoder collapse |
| `run_20260424_000149` | abs + max (old) | 0.571 | bug 1 fixed, bug 2 remained |
| `run_verify_default`  | **rel + mean** (new default) | **0.819** | bug 2 fixed |

Per-task trajectory AUROC after both fixes: Bread 0.81 / Cereal 0.97 / Milk 0.89.
Reproduced old behavior with `SCORE_MODE=abs TRAJ_SCORE_AGGREGATOR=max` → 0.57
pooled (backward compat intact). Not yet at LPB's 0.97 pooled, because LPB uses
KNN retrieval on features rather than transition error; closing the remaining
gap is the job of Phase-B + `ω>0` CFG (not investigated in this session).

## 1. Bug 1: encoder representation collapse

### Symptom
`phase_A` training log showed `latent_mse ≈ 1e-6` and `held_out_r_plus ≈ 5e-4`
from epoch 1. Looked "too good"; eval pooled AUROC was 0.50.

### Root cause
`scripts/train_d4.sh` defaulted `ENCODER_FREEZE=0`, `ENCODER_PRETRAINED=0`, no
`ENCODER_CHECKPOINT`. Combined with `residual_latent_head=True`
(`pred = z_t + Δ`) and `loss = MSE(pred, z_target.detach())`, the easiest way
to drive loss to zero is to collapse all embeddings — then `z_t ≈ z_target`
and Δ=0 trivially satisfies the objective. Classic BYOL-style collapse when
a shared encoder is trained with a stop-gradient target but no momentum
teacher / predictor constraint.

### Evidence (trained encoder)
5 frames from a PickPlaceBread trajectory through the saved encoder:
```
||z|| ≈ 45.2 (all 5 identical to 2 decimals)
per-dim std across 5 frames = 0.0044
pairwise L2 distances = 0.10 ~ 0.29  (should be O(||z||))
```

### Fix
`scripts/train_d4.sh` defaults changed to mirror
`lpb/scripts/train_lpb_dynamics.sh`:
```
ENCODER_FREEZE=1
ENCODER_PRETRAINED=1
ENCODER_CHECKPOINT=${REPO_ROOT}/robosuite/RL/models/resnet18-f37072fd.pth
```

After retraining: `latent_mse ≈ 0.075/elem`, `held_out_r_plus ≈ 39`, no
collapse. But eval pooled AUROC only rose from 0.50 → 0.57 — exposing bug 2.

## 2. Bug 2: motion-magnitude confound in per-step score

### Symptom (after bug 1 fix)
- Per-frame AUROC reasonable (Bread 0.81 / Cereal 0.85 / Milk 0.79).
- Trajectory AUROC collapses (Bread 0.66 / Cereal 0.51 / **Milk 0.48**,
  i.e. *inverted*).

### Root cause
Detector score per `d4disc_0423.md §8.2`:
```
λ_t = ||f_ω − z_target||²  /  (2σ²)
```
With `residual_latent_head=True`, `f_+ = z_t + Δ`, so
```
||f_+ − z_target||² = ||Δ − (z_target − z_t)||²
                    = ||Δ − d_true||²
```
When Δ is only a modest correction (Phase-A-only training, frozen encoder,
per-elem MSE 0.075), this quantity is still **dominated by
`||z_target − z_t||²` — i.e. per-step motion magnitude** — not by how well
the model predicted the delta.

Per-task motion statistics then confound the score:

| task | id_err mean (succ) | id_err mean (fail) | pred_err mean (succ) | pred_err mean (fail) |
|---|---|---|---|---|
| Bread | 43.9 | 48.9 | 33.0 | 42.1 |
| Cereal | — | — | — | — |
| Milk  | 29.8 | **20.1** | 23.3 | **17.2** |

PickPlaceMilk failures stall more than successes → smaller motion → smaller
absolute residual → AUROC inverts.

### Quantitative sweep (same ckpt, 30+30 per task)

| score form | Bread | Cereal | Milk | POOLED |
|---|---|---|---|---|
| abs, raw_mean (old) | 0.65 | 0.81 | **0.12** | 0.52 |
| abs, raw_max (old)  | 0.76 | 0.66 | 0.43 | 0.61 |
| abs, prefix_mean+max (shipped default) | 0.63 | 0.59 | 0.23 | 0.50 |
| **rel, raw_mean (new)** | **0.98** | **0.94** | **0.98** | **0.89** |
| ratio (pred/id) | 0.37 | 0.75 | 0.52 | 0.54 |

### Fix
Change the per-step score to subtract the identity baseline:
```
λ_rel_t = (||f_ω − z_target||² − ||z_t − z_target||²) / (2σ²)
```
Interpretation: model's NLL *improvement over the "no-model" prior*
(`pred = z_t`). Negative on ID (model beats identity), ≈ 0 on OOD
(model no better than identity).

Theoretically consistent with the residual-head form: the trained Δ *is*
the delta predictor, and its RMS error is the right OOD signal. `abs` form
is kept as an ablation (matches `d4disc_0423.md §8.2` verbatim).

### Files changed (bug 2)
1. `robosuite/discriminator/d4disc/inference/detector.py`
   - `D4Detector.__init__` adds `score_mode: str = "rel"`, validated against
     `{"rel", "abs"}`.
   - `_score_frames` subtracts `id_err` when `score_mode == "rel"`.
   - Class docstring explains the motion-confound rationale.
2. `robosuite/discriminator/d4disc/inference/benchmark.py`
   - `D4BenchmarkDiscriminator` accepts & forwards `score_mode`, records it
     in `calibration_summary`.
3. `data/utils/benchmark/examples/run_d4.py`
   - New `--score-mode {rel,abs}` CLI flag, default `rel`.
4. `robosuite/discriminator/d4disc/scripts/run_d4_benchmark.sh`
   - `SCORE_MODE=${SCORE_MODE:-rel}`, threaded through to `--score-mode`.
   - `TRAJ_SCORE_AGGREGATOR` default: `max` → `mean` (the pre-existing
     comment in `run_d4.py` had already flagged that D4 pos-only signal is
     too weak for `max` — this makes the default self-consistent).

### Not changed
- `D4Trainer` / `compute_advantage_gate` — training target is still
  Gaussian-NLL abs MSE, which is the right learning objective. Only the
  *inference scoring* is motion-adjusted.
- `D4Detector.calibrate` — τ is a δ-percentile of λ, works unchanged for
  negative λ.
- `residual_latent_head` stays `True` (Phase-A stability fix from
  `d4disc_v2_bench_and_ckpt_bugs.md §4`).

## 3. Invariants surfaced for future debugging

1. **Residual head + shared encoder + modest training** ⇒ the raw residual
   score is dominated by motion magnitude. For any future CFG variant,
   either (a) subtract identity baseline at inference, or (b) train Δ to
   near-zero error (full Phase-B usually does this).
2. **Phase-A collapse trap**: if `latent_mse` drops below ~1e-4 per element
   in the first 1-2 epochs, suspect encoder collapse before celebrating.
   Gate: with `residual_latent_head=True` and frozen ImageNet encoder,
   expect per-element MSE to land around 5e-2 to 1e-1 (motion-magnitude
   scale on 128×128 cached frames).
3. **Eval aggregator coupling**: `LAMBDA_MODE=mean, LAMBDA_WINDOW=-1`
   (cumulative prefix mean) + `TRAJ_SCORE_AGGREGATOR=max` is the
   *worst* combo for weak per-step signal — `max` picks up early-prefix
   noise. Prefer `mean` aggregator for D4 until Phase-B + ω>0 sharpens
   the per-step gap.
4. **Task-motion asymmetry is real**: PickPlaceMilk failures have *less*
   motion than successes (stall-mode failure), so any absolute-error OOD
   score inverts on this task. Any new score design must be motion-
   invariant to avoid this trap.

## 4. How to reproduce / regress

```bash
# Default (both fixes active)
bash robosuite/discriminator/d4disc/scripts/run_d4_benchmark.sh
# → pooled ~0.82

# Old pre-fix behavior (for regression)
SCORE_MODE=abs TRAJ_SCORE_AGGREGATOR=max \
  bash robosuite/discriminator/d4disc/scripts/run_d4_benchmark.sh
# → pooled ~0.57

# Retrain phase-A baseline with fixed encoder defaults
SKIP_BOOTSTRAP=1 WARM_UP_EPOCHS=10 \
  bash robosuite/discriminator/d4disc/scripts/train_d4.sh \
    --run-name d4_lpb_parity_phaseA \
    --save-dir checkpoints/d4disc/dynamics/d4_lpb_parity_phaseA
```

## 5. Open items (not addressed this session)
- Phase-B + ω>0 to close the residual gap to LPB's 0.97 pooled.
- Re-check that the Phase-B advantage gate (which still uses raw
  `(r_minus - r_plus)` in `training/filter.py`) is not similarly
  motion-confounded; under residual head, the difference cancels the
  `||d_true||²` term analytically, so likely fine, but worth verifying on
  the new non-collapsed encoder.
- Consider whether `d4disc_0423.md §8.2` should be updated to document
  `score_mode="rel"` as the default; currently the design doc still
  describes only the `abs` form.
