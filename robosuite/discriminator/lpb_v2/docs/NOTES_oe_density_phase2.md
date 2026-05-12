# OE-Density Discriminator — Phase 2 Implementation Notes

Working log for the OE-density score discriminator that lives in
`robosuite/discriminator/lpb_v2/`. Companion to `PLAN_score_based_discriminator.md`
(theory) and `PROMPT_oe_density.md` (original implementation spec). This file
records what was actually built, what broke, and what was changed — read it
before re-running or extending.

Author: working notes, 2026-05-12.

---

## 1. Context

- Target metric: **frame-level** failure detection on real-world AgileX (sim is saturated).
- Baselines on the same eval split:
  - Single-bank LPB v2 KNN: pooled frame AUROC = **0.716** (`real_world_eval/run_20260429_103014`).
  - Two-bank KNN: pooled frame AUROC = **0.832** (`two_bank_eval/run_20260512_164541_two_bank_diff_a1`).
  - Pooled Mahalanobis on raw z (Phase 1): frame AUROC = 0.74 (`latent_separability/run_20260512_155300`).
- Phase 2 target: **≥ 0.882** pooled frame AUROC (+5 pp over two-bank), no per-task regression > 3 pp.
- WAM latent is frozen; only a learned score head on top of `LPBV2BenchmarkDiscriminator._encode`.

## 2. What was built

| File | Role |
|---|---|
| `models/score_oe_density.py` | `OEDensityScore` nn.Module — linear projection `f` + Cholesky-Gaussian `p_θ` + joint loss. |
| `benchmark_oe_density.py` | `OEDensityBenchmarkDiscriminator` — subclasses `LPBV2BenchmarkDiscriminator`, reuses `_encode`, pools across tasks with task one-hot, per-task threshold calibration. |
| `run_oe_density_real_world_benchmark.py` | CLI mirroring the two-bank CLI. |
| `scripts/run_oe_density_real_world_benchmark.sh` | env-var wrapper. |
| `tests/test_score_oe_density.py` | 6 unit tests (synthetic planted direction, λ=0 gradient regression, state-dict round-trip, disjointness assertion, first_gt-None skip). |

Public API additions in `lpb_v2/__init__.py`: `OEDensityScore`, `OEDensityBenchmarkDiscriminator`.

## 3. First run — broken (do not use)

Run dir: `checkpoints/lpb_v2/oe_density_eval/run_20260512_205026_oe_density`.

Results: pooled frame AUROC = **0.570** (worse than every baseline). Trajectory AUROC = 0.794, detection delay ≈ −127 frames (predictions fire ~130 frames before `first_gt_failure_frame`).

**Root cause** (from `ftrain_manifest.json:train_history`): `L_fail = 0` from epoch 1 to 199. The `λ=1` run was effectively `λ=0` for 99% of training.

The original PROMPT specified the hinge form
```
L_fail = λ · max(0, log p_θ(f(z_fail)) − m)
m = quantile_0.05( log p_θ(f(z_succ_calib)) )      # recomputed each epoch
```
This saturates to 0 after one or two epochs:
1. The 16-D random projection at init creates an artificial gap between `f(z_succ)` and `f(z_fail)` in density space (curse of dimensionality acts at init).
2. As `L_succ` tightens `Σ` around `μ_emp(succ)`, `log p_succ` rises monotonically, and `m` (its 5%-quantile) rises in lockstep.
3. Fail-side log-density does *not* rise (no gradient is pushing it up either), so `log p_fail < m` for every fail sample after epoch 1 — hinge gives zero gradient forever.
4. From epoch 2 onward, the joint loss is exactly `−mean log p_succ + γ ||W1||²` — Gaussian density on an unsupervised PCA-like projection of success. The result is structurally identical to the failure mode the plan §3.3 already diagnosed (PCA captures task/view variance, not failure-discriminative variance).

Trajectory-AUC=0.79 + frame-AUC=0.57 + detection-delay=−127 is the unmistakable fingerprint of *one-class density on the wrong subspace*: failure trajectories have slightly lower aggregate density, but within a failure trajectory the model can't tell which frame is the failure.

## 4. Fix applied (2026-05-12)

Replaced the dying hinge with the Hendrycks-OE **anti-density penalty**:
```
L_fail = λ · mean( log p_θ(f(z_fail)) )
```
Always carries gradient. Combined with `L_succ = −mean log p_θ(f(z_succ))`, this is exactly *log-likelihood-ratio maximisation*.

**Scale-invariance pathology found and fixed.** With Gaussian `p_θ`, rescaling `W1 → α·W1` shifts `L_total` by `(1−λ) · k · log α`. At `λ = 1` this is 0 — the loss has a flat direction along which `||W1|| → ∞` and `σ → 0` together (verified empirically: |W1| 0.98 → 49, σ 1.01 → 1e-4, cos with planted direction collapsed to 0.27). At `λ < 1` divergence is just slower. **Fix:** L2-normalise rows of `W1` inside `forward()` so the projection scale is fixed and `Σ` (Cholesky) carries the only meaningful scale. With normalisation:
- Synthetic test (k=1, shift 4σ): cos ≈ 0.997, held-out AUROC ≈ 0.998 across `λ ∈ {0.5, 1.0, 2.0}`.

**Numerical floor.** Bumped Cholesky-diagonal floor from `1e-4` to `1e-2` in `_build_L` (and matched in `init_density_from_batch`) so `σ` cannot collapse to FP-noise on real data.

**Bookkeeping removed:**
- `OEDensityScore.set_margin` and the `margin` buffer (no margin in the new loss).
- Per-epoch `m_val = torch.quantile(log_p_calib, ...)` recomputation in the adapter.
- `--margin-quantile` CLI flag and `MARGIN_QUANTILE` env var.
- `margin_m` / `margin_m_final` keys from `train_history` and `calibration_summary`.

**Diagnostic additions:**
- `train_history` rows now log `log_p_succ_mean` (derived from `−L_succ`) and `log_p_fail_mean` (derived from `L_fail / λ`).
- `calibration_summary` includes `fail_loss_form: "anti_density"` and final mean log-densities for per-task entries.

All 6 unit tests pass.

## 4b. Second run — diverged (do not use)

Run dir: `checkpoints/lpb_v2/oe_density_eval/run_20260512_211335_oe_density`.

Results: pooled frame AUROC = **0.796** (better than single-bank KNN 0.716, worse than two-bank 0.832, below the 0.882 target). Trajectory AUROC = 0.764. **Loss diverged monotonically** to `L_total ≈ -2.0e21` by epoch 199 — but the Mahalanobis ranking partially survived, hence the non-trivial AUROC.

**Root cause.** The anti-density loss at λ=1 is `L_total = -E_s log p + E_f log p = log(p_fail / p_succ)`. Under a Gaussian, this is **mathematically unbounded below**: shift `µ → µ_succ + t·(µ_succ - µ_fail)` and `q_succ - q_fail → -2t·‖µ_succ - µ_fail‖²_Σ → -∞`. The W1-row-normalisation in §4 blocked the W1-scale flat direction but left µ unconstrained. The synthetic test (k=1, 4σ planted shift, init_density anchored) doesn't exercise this regime; with k=16 on the real WAM latent there's enough geometric freedom for µ to drift.

Evidence in `train_history`: L_succ grows from 301 → 2.15e21, L_fail from -944 → -4.17e21, ratio L_fail/L_succ ≈ -2 throughout (µ riding the discriminative axis). No saturation, no convergence.

## 4c. Fix applied (2026-05-12, second pass)

**Replaced the anti-density loss with a logistic discriminative loss** on the score `s(z) = -log p_θ(f(z))`:

```
L_succ = mean softplus( s(z_s) - τ )         # wants s_succ < τ
L_fail = λ · mean softplus( τ - s(z_f) )     # wants s_fail > τ
L_reg  = weight_decay · ‖W1‖²_F
L_total = L_succ + L_fail + L_reg
```

with `τ` a learnable scalar threshold initialised at the median of `s(z_succ_train)` (puts the loss in its linear regime at start). Each softplus is bounded below by 0 and saturates as samples are correctly classified, so the loss can no longer escape to -∞. This is exactly the PROMPT §7 / PLAN §4.J #2 fallback (low-rank logistic on `f`), with the same (W1, b1, µ, L_raw) parameterisation so the test-time `score()` and the per-task adapter calibration are unchanged.

**Bookkeeping changes:**
- `OEDensityScore` gains a learnable `tau` Parameter (added to optimizer param groups, included in state-dict).
- `init_density_from_batch` additionally sets `tau ← median(s(z_succ))`.
- `loss()` returns dict keys `score_succ_mean, score_fail_mean, tau` (replacing the old `log_p_succ_mean / log_p_fail_mean` derivations).
- `train_history` rows now log `score_succ_mean, score_fail_mean, tau` per epoch; `[oe_density][train]` print line shows `s_succ, s_fail, tau, gap`.
- `calibration_summary` per-task entries: `fail_loss_form: "logistic"`, `score_{succ,fail}_mean_final`, `tau_logistic_final`. Top-level: `fail_loss_form: "logistic"`.

**Disambiguation:** there are now two `tau` symbols in the system. `OEDensityScore.tau` is the logistic-regression threshold learned during training. `OEDensityBenchmarkDiscriminator._tau_per_task[task]` is the per-task percentile/Youden threshold calibrated post-hoc on succ scores (used at test time for the `score >= τ` binary prediction). These are not coupled.

**Tests:** added `test_loss_stays_bounded_under_long_training` (regression guard against any future reintroduction of an unbounded loss form: |L_total| < 1e3 for 500 steps on 8-D toy data). All 7 tests pass.

## 5. Stale docs

`PROMPT_oe_density.md` §3.3–§3.4 and `PLAN_score_based_discriminator.md` §4.C–§4.D describe the dying-hinge form; §4.A–§4.B describe the anti-density form. Both are superseded by the logistic loss in §4c above. The module's top docstring (`models/score_oe_density.py`) records the revision inline. Update those docs only if Phase 2 actually lands the AUROC target and the method becomes the committed baseline.

## 6. Open questions / next steps

1. **End-to-end rerun on real-world AgileX** with the logistic loss. Target ≥ 0.882 pooled frame AUROC. Run via `bash scripts/run_oe_density_real_world_benchmark.sh` (defaults: `λ = 1.0`, `score_k = 16`, `num_epochs = 200`, logistic form, W1 row-normalised, learnable `τ`).
2. **Explicit `λ = 0` ablation** (`LAMBDA=0`) — the PROMPT §7 comparator. At λ=0 only the succ-side softplus is active; the fail term is bit-identical to the None case (guarded by `test_lam_zero_ignores_fail_data_bitwise`). Expect a clean gap vs λ>0 because the projection is no longer pulled by fail samples.
3. **Sanity in `train_history`:** confirm both softplus components stay bounded, `score_fail_mean - score_succ_mean` grows positive and stabilises, and `tau` lands between the two means. If the gap stays flat near 0, the model isn't learning a discriminative direction and the diagnosis is different.
4. **If the AUROC target is missed by < 5 pp:** sweep `λ ∈ {0.3, 0.5, 1.0, 2.0}` and `score_k ∈ {8, 16, 32}`. With the bounded logistic loss, `λ > 1` simply upweights the fail term relative to succ, no instability.
5. **If the AUROC target is missed by ≥ 5 pp:** the logistic form on a quadratic score is *already* the §4.J #2 fallback. Next step is candidate B (deep one-class metric + two-bank KNN in `f`-space), or replacing the Gaussian score with a small MLP head (Phase 3 escalation).

## 7. Files of record

- Theory: `robosuite/discriminator/lpb_v2/PLAN_score_based_discriminator.md`
- Original implementation spec: `robosuite/discriminator/lpb_v2/PROMPT_oe_density.md` (parts §3.3/§3.4 now stale).
- This notes file: `robosuite/discriminator/lpb_v2/NOTES_oe_density_phase2.md`
- Broken first run (dying hinge): `checkpoints/lpb_v2/oe_density_eval/run_20260512_205026_oe_density`
- Diverged second run (anti-density, λ=1 unbounded): `checkpoints/lpb_v2/oe_density_eval/run_20260512_211335_oe_density`
- Two-bank baseline (yardstick): `checkpoints/lpb_v2/two_bank_eval/run_20260512_164541_two_bank_diff_a1`
- Phase 1 diagnostic: `checkpoints/lpb_v2/latent_separability/run_20260512_155300`
