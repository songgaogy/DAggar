# Score-Based Frame-Level Failure Discriminator on the WAM Latent

A focused research plan that picks one direction out of the broader brainstorm
in `robosuite/discriminator/IDEAS_failure_supervision.md` and works it out to
the point where it can be implemented. This document is the **chosen plan**;
the broader notes remain as context.

Target metric: **frame-level** failure detection on real-world AgileX
trajectories (and robosuite for sanity), with theoretically grounded
calibration. Sequential / temporal aggregation tricks are explicitly out of
scope.

---

## 1. Context (one paragraph)

The current `LPBV2KNN` builds a memory bank from success demos and predicts
failure when a frame's WAM latent is more than `tau` away (L2, percentile-
calibrated) from the bank. The visualization in
`checkpoints/lpb_v2/real_world_eval/run_20260511_170606_vis/` shows that the
WAM latent likely encodes useful structure: `success` and `failure_after_gt`
overlap in PCA but separate locally in t-SNE. KNN works in this regime —
locally-separable, globally-overlapping — but the convergence rate of KNN as a
density estimator is `O(N^{-1/d})` with `d ≈ 551`, which is the root cause of
its plateau in real-world settings: irrelevant variance in the latent drowns
out the failure direction in plain L2. The next step is to replace KNN with a
score-based method that is (a) theoretically grounded and (b) data-efficient
in the failure-label regime.

## 2. Core theoretical framing

We make a strict separation between three theoretical primitives:

1. **Representation**: the frozen WAM latent `z ∈ R^D` (D ≈ 551). Not touched.
2. **Score function**: a learned `s: R^D → R` with explicit probabilistic
   meaning (log-density or its monotonic transform).
3. **Decision rule**: a calibrated threshold with **finite-sample, distribution-
   free** type-I error control, via split conformal prediction.

The plan: a **supervised 1-dimensional projection** that is the
Bayes-sufficient statistic for the success-vs-failure classification problem
under a Gaussian model, followed by **1-d density estimation** in the
projected space, followed by **conformal calibration**.

Why this is not "naive":

- KNN is the plug-in estimator for `p(z)`. Its `O(N^{-1/d})` rate breaks at
  d ≈ 551. A 1-d density estimator on a discriminative projection has rate
  `O(N^{-4/5})` (1-d KDE) — an order-of-magnitude improvement in sample
  efficiency.
- The projection uses **D parameters** (one direction) closed-form; it is
  data-efficient even with a few thousand failure frames, which matters
  because failure data is scarce.
- The decision threshold has a **finite-sample type-I bound** via conformal,
  not a heuristic percentile.

## 3. Phase 1 — Diagnostic: does the WAM latent carry frame-level signal at all?

Before any modeling, verify the premise: that success-frame and
`failure_after_gt`-frame distributions differ at the **frame level**, not just
the trajectory level. Visual inspection of t-SNE is not sufficient; we want
statistical evidence with controls.

### 3.1 Two-sample test

For each task `T ∈ {candy_in_plate, duck_in_bowl, Micky_in_box, sausage_in_pot}`:

- Sample success frames `Z^s_T` and `failure_after_gt` frames `Z^f_T`.
- Compute `MMD²(Z^s_T, Z^f_T)` with a Gaussian kernel; bandwidth via the
  median heuristic.
- Permutation test (≥ 1000 permutations) for an exact finite-sample p-value
  under `H₀: P_{success-frame} = P_{failure_after_gt-frame}`.

**Decision rule**

- If all four tasks give `p < 0.01`: distributions are statistically different.
  Proceed to §3.2.
- If any task gives `p > 0.05`: the latent has no frame-level signal on that
  task. The entire plan does not apply there; fix the representation first.

### 3.2 Effect size: per-frame Mahalanobis AUROC

A p-value answers "is there a difference"; AUROC answers "how strong is it":

1. Fit `N(μ, Σ_LW)` on success training frames, where `Σ_LW` is the
   Ledoit-Wolf shrinkage covariance (necessary at D ≈ 551).
2. Per-frame score: `s(z) = (z − μ)ᵀ Σ_LW⁻¹ (z − μ)`.
3. Compute frame-level AUROC on held-out `(success, failure_after_gt)` frames.
4. **Split by rollout, not by frame**, to avoid leakage from temporally-
   adjacent correlated frames.

**Decision rule**

- AUROC > 0.75: strong signal, the plan has clear headroom over KNN.
- 0.60 < AUROC < 0.75: marginal — requires the supervised projection from §4
  to surface the signal.
- AUROC < 0.60: representation is the bottleneck. Stop and address that first;
  do not proceed.

### 3.3 Confound controls (mandatory)

Two confounds will inflate §3.1 and §3.2 if uncontrolled:

- **Task confound**. `failure_after_gt` may concentrate in particular tasks;
  measuring "success vs failure" can devolve into "task A vs task B". **All
  tests are per-task.** Aggregate across tasks via harmonic mean (or report
  per-task numbers directly).
- **Timestep confound**. `failure_after_gt` frames live in the late portion of
  rollouts. If the WAM latent encodes "episode phase", late success frames will
  also look failure-like. Add a **matched-timestep test**: for each
  `failure_after_gt` frame at `(task=T, t=k)`, compare against success frames
  in the same task with `t ∈ [k − 5, k + 5]`.

If either confound dominates, the headline numbers must be replaced by the
controlled numbers.

## 4. Phase 2 — Supervised projection: Fisher LDA with Ledoit-Wolf

### 4.1 Setup

Assume locally `p(z | success) = N(μ_s, Σ)` and `p(z | failure) = N(μ_f, Σ)`
(shared covariance, can be relaxed). The Bayes-optimal classification
direction is

```
w* = Σ⁻¹ (μ_f − μ_s)
```

and `f(z) = w*ᵀ z ∈ R` is the **1-d sufficient statistic** for the
classification problem (data-processing inequality holds with equality at a
sufficient statistic). Under the Gaussian assumption, no information about the
success/failure classification is lost by reducing the 551-d latent to this
one number.

### 4.2 Why this beats MLP / contrastive in our regime

| Property                            | Fisher | 2-layer MLP head | Supervised contrastive |
|------------------------------------|--------|------------------|------------------------|
| Parameters                          | D      | ~10⁵             | ~10⁴                   |
| Closed-form / no optimization noise | Yes    | No               | No                     |
| Bayes-optimal under Gaussian        | Yes    | Asymptotic       | No (different objective)|
| Stability with N_fail ≈ 3 000       | High   | Overfits         | Overfits / temperature- sensitive |

Fisher uses the smallest model that is information-theoretically sufficient.
With limited failure data, this matters more than expressive capacity.

### 4.3 Required ingredient: Ledoit-Wolf shrinkage covariance

At D ≈ 551 with N ≈ 10⁴–10⁵, the raw sample covariance is unstable and often
numerically singular. Ledoit-Wolf (2004) gives a closed-form optimal shrinkage:

```
Σ_LW = (1 − α*) Σ̂ + α* σ̄² I,   α* derived analytically from data.
```

Provably minimum-MSE relative to the true covariance under high-dimensional
asymptotics. This is **not an engineering trick** — without it, Fisher is
ill-defined here.

### 4.4 Per-task vs pooled direction

Open question, decide empirically:

- **Pooled `w*`** across all tasks: more samples, lower variance, but assumes
  the failure direction is shared across tasks.
- **Per-task `w*_T`**: each task gets ~750 failure frames — borderline for
  estimating `Σ⁻¹` even with shrinkage; but captures task-specific failure
  modes.

Initial recommendation: pooled, with per-task evaluation. Re-evaluate after
Phase 1 results show task heterogeneity.

### 4.5 Relaxations

- **Heteroscedastic**: drop the shared-Σ assumption, use QDA. Quadratic in z;
  more parameters, more failure data required.
- **Non-Gaussian**: replace Fisher with logistic regression on `(success,
  failure_after_gt)`. Asymptotically equivalent to Fisher under Gaussian;
  more robust to deviations. Same parameter count. **Preferred fallback** if
  the Gaussian assumption visibly breaks (check via §5 diagnostics).

## 5. Phase 3 — 1-d density and conformal threshold

### 5.1 Density in the projected space

In `f(z) ∈ R`, fit `p̂_{succ}(·)` on the **success calibration split** (disjoint
from §4's training data):

- Simplest: 1-d Gaussian `N(m, s²)`.
- More flexible: 1-d KDE (Silverman bandwidth) — converges at `O(N^{-4/5})`,
  effectively free in 1-d.
- Most flexible: 1-d normalizing flow (overkill in 1-d but available).

Per-frame anomaly score: `score(z) = − log p̂_{succ}(f(z))`.

### 5.2 Decision rule via split conformal

Given a desired false-alarm level α:

1. Compute scores `{score(z_i)}` on a **success calibration set** disjoint from
   §4 and §5.1.
2. Threshold `τ = (1 − α)`-quantile of the calibration scores.
3. Predict failure when `score(z_test) > τ`.

**Theorem (Vovk, split conformal)**: for any α and any score function,

```
P(score(z_new) > τ) ≤ α + 1 / (n_cal + 1)
```

under exchangeability of the calibration and test points within the success
class. This is the finite-sample type-I bound; nothing else in the pipeline
gives this.

### 5.3 Why this beats the current KNN at theoretical level

- **Convergence rate**: 1-d KDE `O(N^{-4/5})` vs 551-d KNN `O(N^{-1/551})`.
  Orders of magnitude better sample efficiency.
- **Score units**: KNN distance is bank-dependent; `− log p̂` is intrinsic and
  cross-experiment comparable.
- **Calibration**: percentile-on-success is a heuristic; conformal is a
  theorem.
- **Failure information**: KNN ignores failure data entirely; here failure
  data is used to define `w*`, but **not** to estimate `p̂_{succ}` and **not**
  to set `τ` — so the conformal exchangeability assumption is preserved.

## 6. Positioning vs existing baselines

All baselines (`float/`, `logpZO/`, `lpb/`, current `lpb_v2/` KNN) are
success-only. Our method **does** use failure data, but only to determine the
Bayes-optimal 1-d projection direction; the density and threshold are still
success-only. This is the minimal departure from success-only and is the
cleanest fairness story we can offer.

Two natural ablations to run alongside:

- **No Fisher (success-only)**: replace `f(z)` with PCA's top component →
  measure the lift attributable to using failure data for the projection.
- **No conformal**: keep Fisher + 1-d density, but threshold by the original
  percentile-on-success rule → measure the lift attributable to conformal.

## 7. TODOs

Order matters: §7.1 gates everything else.

### 7.1 Phase 1 diagnostic (gating)

- [ ] `utils/diagnose_latent_separability.py` — script that:
  - Loads `LPBV2Encoder`, encodes frames from both classes via the existing
    `LPBV2BenchmarkDiscriminator` plumbing.
  - Runs per-task MMD² permutation test (1000 perms, Gaussian kernel, median
    bandwidth).
  - Fits Ledoit-Wolf success Gaussian, reports per-task and pooled
    Mahalanobis AUROC with rollout-level splits.
  - Runs the matched-timestep control.
  - Emits a single JSON summary + per-task PNG ROC curves.
- [ ] Decision gate: does the latent pass §3.1 and §3.2?

### 7.2 Phase 2 implementation

- [ ] `models/fisher_projector.py` — closed-form Fisher direction with
  Ledoit-Wolf shrinkage; pooled and per-task variants.
- [ ] `models/logistic_projector.py` — logistic regression fallback for the
  non-Gaussian case.
- [ ] Unit test: verify on synthetic Gaussian data that the recovered
  direction matches the analytical optimum.

### 7.3 Phase 3 implementation

- [ ] `models/score_density.py` — 1-d Gaussian and 1-d KDE wrappers exposing
  a `log_prob(f)` interface.
- [ ] `discriminators/score_discriminator.py` — composes
  `LPBV2Encoder + FisherProjector + ScoreDensity + ConformalThreshold`.
- [ ] Integrate as a new `BenchmarkDiscriminator` so `FailureBenchmark` plumbing
  is reused unchanged.

### 7.4 Evaluation

- [ ] Run on robosuite (sanity, must not regress).
- [ ] Run on real-world AgileX (the target).
- [ ] Ablations: (a) PCA direction instead of Fisher; (b) percentile threshold
  instead of conformal; (c) per-task vs pooled `w*`.
- [ ] Compare frame-level AUROC, F1 at target α, and false-alarm-rate
  empirical coverage vs the conformal α bound.

## 8. Open questions to revisit

1. **Per-task vs pooled `w*`** — to decide empirically after Phase 1 reveals
   how heterogeneous failure modes are across the four tasks.
2. **Gaussian vs logistic projector** — switch if a `(f(z_succ), f(z_fail))`
   1-d histogram visibly violates Gaussian (heavy tails, multi-modality).
3. **Multi-dimensional projection** — only if 1-d Fisher AUROC underperforms
   the raw Mahalanobis AUROC by more than 5 points, which would falsify the
   Gaussian + shared-Σ assumption and motivate moving to kernel Fisher /
   sliced inverse regression.
4. **Fairness narrative** — the method uses failure data; baselines do not.
   The "no Fisher (PCA direction)" ablation in §6 is the way to disentangle
   "score-based density helps" from "failure-data projection helps".

## 9. References

- Vovk, Shafer, Gammerman. *Algorithmic Learning in a Random World*, 2005.
  Split conformal prediction; finite-sample coverage.
- Gretton et al. *A Kernel Two-Sample Test*, JMLR 2012. MMD + permutation test.
- Ledoit, Wolf. *A well-conditioned estimator for large-dimensional covariance
  matrices*, J. Multivariate Analysis 2004. Shrinkage covariance.
- Fisher. *The use of multiple measurements in taxonomic problems*, 1936.
- Lee et al. *A Simple Unified Framework for Detecting OOD Samples and
  Adversarial Attacks*, NeurIPS 2018. Mahalanobis OOD detection on deep
  features.
- Kirichenko, Izmailov, Wilson. *Why Normalizing Flows Fail to Detect OOD
  Data*, NeurIPS 2020. (Relevant if extending §5.1 from 1-d to flow on
  reduced latent.)

---

*Living document. Author: working notes, 2026-05-12. Companion to the broader
brainstorm at `robosuite/discriminator/IDEAS_failure_supervision.md`.*
