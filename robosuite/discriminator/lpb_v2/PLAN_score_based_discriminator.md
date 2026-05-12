# Score-Based Frame-Level Failure Discriminator on the WAM Latent
A focused research plan picking one direction from `robosuite/discriminator/IDEAS_failure_supervision.md` and committing to a **learning-based** score function `s: R^D → R` on top of the frozen WAM latent. The two-bank KNN baseline (zero-parameter, see `PROMPT_two_bank_knn.md` and `checkpoints/lpb_v2/two_bank_eval/run_20260512_164541_two_bank_diff_a1`) is a yardstick this method must clearly beat; it is **not** the destination.

Target metric: **frame-level** failure detection on real-world AgileX trajectories (sim is saturated). Sequential / temporal aggregation tricks are explicitly out of scope.

---

## 1. Context (one paragraph)
The current `LPBV2KNN` builds a memory bank from success demos and predicts failure when a frame's WAM latent `z ∈ R^D` (D ≈ 551) is more than `tau` away (L2, percentile-calibrated) from the bank. PCA overlaps but t-SNE separates the success vs `failure_after_gt` distributions (`checkpoints/lpb_v2/real_world_eval/run_20260511_170606_vis/`) — the regime KNN is *suited* for in principle, but KNN's `O(N^{−1/d})` rate at d ≈ 551 means irrelevant variance in plain L2 drowns out the failure direction. Two recent results bound the design space:

- **Phase 1 diagnostic** (`checkpoints/lpb_v2/latent_separability/run_20260512_155300`): pooled Mahalanobis frame AUROC = **0.74** (gate `proceed_to_phase_2_marginal`), matched-timestep AUROC = 0.70 (signal survives the late-frame confound), per-task spread 0.66–0.79 (13pp).
- **Two-bank KNN baseline** (`...two_bank_eval/run_20260512_164541_two_bank_diff_a1`): pooled frame AUROC = **0.832** (+11.6pp over the 0.716 single-bank LPB v2 KNN), built by adding a held-out fail-reference bank to plain-L2 KNN. Per-task heterogeneity persists (one task improves much less than the others); we note this as a data point about per-task variance but **do not chase it as a separate research thread**.

The goal of this plan is a **learning-based score function** `s(z) ∈ R` that consumes a small held-out set of gt-fail trajectories (10/task, disjoint from eval and from the two-bank reference) and beats two-bank KNN's 0.832 by a margin large enough to justify the added complexity. **Target**: ≥ +5pp pooled frame AUROC over two-bank, with no per-task regression > 3pp.

## 2. Core theoretical framing
Three primitives, with one critical asymmetry vs the earlier draft of this plan: **we estimate density only on the success side, and use the small gt-fail set as constraint points, not as samples of `p_fail`.**

1. **Representation**: frozen WAM latent `z ∈ R^D` (D ≈ 551). Not touched.
2. **Score function**: a learned `s(z): R^D → R` defined as an **outlier-exposed (OE) success-side likelihood**:
   ```
   f(z) = W₁ z + b₁     ∈ R^k       (linear projection, k ≪ D)
   p_θ(·)                          (parametric density on f-space)
   s(z) = - log p_θ(f(z)) ∈ R       (success-side negative log-likelihood)
   ```
   `f` and `p_θ` are trained jointly:
   ```
   L = - E_succ[ log p_θ(f(z)) ]
       + λ · E_fail[ max(0, log p_θ(f(z_fail)) - m) ]    ← hinge on log-density
       + γ · ||W₁||_F²
   ```
   Success frames anchor `p_θ` with low variance (`N_succ` is large). Gt-fail-after-gt frames enter only through a hinge: they contribute gradient only when they sit inside the success bulk. `failure_before_gt` frames are dropped (matches IDEAS §5 safe primitives).
3. **Decision rule**: threshold on `s(z)`. Default split-conformal on a held-out success set (finite-sample type-I bound, prior-free); Youden's J on a third disjoint two-class pool offered as a knob (same plumbing as `TwoBankKNN`'s `--calib-mode two_class_youden`).

Why asymmetric, not discriminative `p(fail|z)/p(succ|z)`:

- **Sample-efficiency.** Direct BCE-fit logistic has boundary variance `~ O(1/√N_fail)` with `N_fail ≈ 2400`, the rare class. OE only relies on fail data to keep a few hinge-violating points outside the success bulk — a few violating points are enough to fix the boundary, so we are not asked to estimate `p_fail`. Most variance is absorbed by `p_θ` fit to large `N_succ`.
- **Prior-odds clean.** Direct logistic learns `log[p(z|fail)/p(z|succ)] + log[p(fail)/p(succ)]`. The training class ratio is artificial (~2400 fail vs much larger succ pool); deployment ratio is task-dependent and unknown. Post-hoc threshold calibration patches the offset but does not fix the logit's shape. The OE score is one-sided likelihood `-log p_θ` — no prior baked in.
- **Why this isn't pure one-class.** Plain `-log p_succ` without fail supervision is what Mahalanobis on raw `z` already does (Phase 1, AUROC 0.74). The novelty is letting fail data shape **the supervised projection `f`** so that `p_θ` lives in a subspace where success and failure are locally separable (the t-SNE-visible regime in `run_20260511_170606_vis`). The fail hinge gives `f` its discriminative gradient signal at near-zero variance cost.
- **KNN's `O(N^{-1/d})` rate breaks at d = 551.** A rank-k linear `f` followed by a parametric `p_θ` is the data-efficient way to *both* align the discriminative subspace *and* estimate density in it.

The earlier "low-rank logistic" (`s = w₂ᵀ f(z) + b₂` trained by BCE) is **kept as a §4.J fallback / ablation**, not as the primary method.

## 3. Phase 1 — Diagnostic [done]
Procedure (MMD permutation test, Ledoit-Wolf Mahalanobis AUROC, matched-timestep confound control) is implemented in `explore/diagnose_latent_separability.py`. Results in `checkpoints/lpb_v2/latent_separability/run_20260512_155300/separability_summary.json`:

- Pooled MMD² permutation p < 0.001 — distributions differ at the frame level.
- Pooled Mahalanobis AUROC = **0.74** — marginal; gate `proceed_to_phase_2_marginal`.
- Matched-timestep AUROC = 0.70 (4pp drop — late-frame confound is small).
- Per-task Mahalanobis AUROC: candy 0.66, sausage 0.75, Micky 0.77, duck 0.79.

Decision: proceed to Phase 2 with a learned projection. Do **not** rely on Mahalanobis alone (it is one of the §4.H baselines).

## 4. Committed design (§4 closed)

The primary method is **OE-density (candidate A)**: linear projection + parametric density on `f(z)` + hinge on fail log-density. §4.A–§4.I commit to it. §4.J lists lower-priority alternatives (low-rank logistic; deep one-class metric learning; NCE) kept either as ablations or as deferred candidates.

### 4.A Projection `f` — linear, `k ≪ D`
- `f(z) = W₁ z + b₁` with `W₁ ∈ R^{k×D}`. Start `k = 16`. Sweep `k ∈ {4, 8, 16, 32}` as an ablation; pick by held-out frame AUROC. Prefer smaller `k` at ties (Occam + cleaner Phase-3 density-form extension).
- **Linear by choice, not laziness.** Parameter count `k·D` ≈ 9k for k=16, vs ~100k for a 2-layer MLP. Pooled gt-fail-after-gt set is ~2400 frames; MLP would overfit. A nonlinear `f` (MLP) is reserved as a §4.H ablation **only after** the linear version is fully exploited.

### 4.B Density `p_θ` — start single full-covariance Gaussian on f(z)
- Default: full-covariance Gaussian fit on `{f(z_succ)}`. Score `s(z) = (f(z) - μ)ᵀ Σ⁻¹ (f(z) - μ)` (Mahalanobis on the projection). `μ, Σ` are differentiable through `W₁`, so the joint loss in §2 trains the projection end-to-end with `p_θ`.
- **Why Gaussian first.** Directly comparable to the Phase 1 Mahalanobis-on-raw-z baseline (AUROC 0.74). The gain over raw Mahalanobis isolates the value of the **supervised projection `f`** — and the gain over `λ=0` OE (same model, no fail supervision) isolates the value of the **hinge signal** (see §4.H).
- Escalation path (only if single Gaussian plateaus): GMM(M=2..4) → KDE on `f(z)` → small normalizing flow on `f(z)`. Each step adds parameters and is justified only by AUROC.
- Ledoit-Wolf shrinkage on `Σ` only if numerical conditioning of the joint optimization needs it; not required by default at k ≤ 32.

### 4.C Hinge margin `m` — data-driven (success log-density quantile)
- `m = quantile_α( log p_θ(f(z_succ_calib)) )` over a held-out success calibration split. α = 0.05 default.
- Interpretation: fail points must sit below the 5th-percentile of success log-density. No human-tuned `m`.
- α is the only knob; sweep `α ∈ {0.01, 0.05, 0.1}` as an ablation. Compute `m` **before** each epoch (not statically) so it tracks `p_θ` as it updates.

### 4.D Hinge weight `λ` — tuned by held-out AUROC; `λ = 0` is the key ablation
- Default `λ = 1.0`. Sweep `{0.1, 0.3, 1.0, 3.0}`.
- **`λ = 0` is the most important ablation in the plan.** It reduces OE to plain Gaussian density on a *supervised* projection of success — but the supervised gradient comes only from `-log p_θ(f(z_succ))`, which is a one-class objective. Comparing `λ = 0` vs `λ > 0` directly measures the value of fail supervision under this score family.
- Practical knob: also report the regularizer weight `γ` on `||W₁||_F²` (start 1e-4); sweep `{1e-5, 1e-4, 1e-3}` if Gaussian fit looks degenerate (`Σ` collapsing along useless directions).

### 4.E Per-task vs pooled — pooled + task one-hot
Phase 1 reports 13pp per-task spread and two-bank reports 17pp; non-negligible. Per-task `f_T` is data-starved (≤ 600 fail-train frames/task) and over-parameterized. Concrete choice: train one pooled `W₁` on `z̃ = [z ; one_hot(task)] ∈ R^{D+T}` with T = 4. Adds 64 parameters, lets each task carry a task-specific bias direction without splitting the data four ways. Density `p_θ` is also pooled in `f(z̃)` space.

### 4.F Calibration
- **Default**: split conformal on success calibration set — finite-sample type-I bound, prior-free, mirrors `LPBV2KNN` / `TwoBankKNN` success-percentile mode.
- **Knob**: Youden's J on a third disjoint fail-calib pool (same plumbing as `TwoBankKNN`'s `--calib-mode two_class_youden`). Use when fail data is plentiful enough to set a useful 2-class threshold.

### 4.G Fairness baselines (mandatory; report all)

| Baseline | Number we have | Source |
|---|---|---|
| Mahalanobis on raw z (Phase 1) | frame AUROC 0.74 | `latent_separability/run_20260512_155300` |
| Single-bank LPB v2 KNN | 0.716 | `real_world_eval/run_20260429_103014` |
| **Two-bank KNN (plain L2)** | **0.832** | `two_bank_eval/run_20260512_164541_...` |
| OE-density `λ=0` (succ-only Gaussian on supervised projection) | TBD | run alongside primary |
| **OE-density `λ>0` linear + Gaussian (§4.A–§4.D)** | **TBD — primary** | this Phase 2 |
| Low-rank logistic (§4.J fallback) | TBD | ablation |
| NCA in place of logistic | TBD | ablation, optional |

The `λ=0` row and the raw-z Mahalanobis row together isolate (i) gain from supervised projection alone, (ii) gain from fail-hinge on top. The logistic row is the head-to-head against the previous primary method.

### 4.H Sanity ablations
- `λ = 0` vs `λ > 0` — does fail supervision help at all? (Most important.)
- Single Gaussian vs GMM(2) vs GMM(4) on `f(z)` — is density form the bottleneck?
- `k ∈ {4, 8, 16, 32}` — does the discriminative subspace have a clear effective rank?
- `α ∈ {0.01, 0.05, 0.1}` (hinge quantile) — boundary sensitivity.
- Pooled-with-task-one-hot vs per-task `f` — is per-task overfit worse than the 13pp spread cost?
- Linear `f` vs 2-layer MLP `f` — last; expected to overfit; reports the ceiling of nonlinear extensions.

### 4.I Out-of-scope (Phase 3 contingencies)
- Replacing `p_θ` with a normalizing flow on f(z).
- Heteroscedastic per-mode Gaussians (needs failure-mode labels we don't have).
- **Hybrid score**: `s_total = α · s_OE + (1-α) · s_two_bank` — justified only if §6.1 (OE-density) and two-bank disagree on different tasks (complementary errors visible in per-task breakdown).

### 4.J Alternative score functions (kept at lower priority)
Three alternatives, in descending priority. None is the main path; each has a well-defined trigger to activate.

1. **Deep one-class metric learning + two-bank KNN (candidate B).** Same linear `f` as §4.A; loss is succ-NN-pull + fail-NN-push hinge instead of OE density:
   ```
   L = E_succ[ ||f(z) - NN_f_succ(f(z))||² ]
     + λ · E_fail[ max(0, m - ||f(z_fail) - NN_f_succ(f(z_fail))||²) ]
   ```
   Scoring reuses the existing two-bank KNN in `f(z)` space — composes structurally with `TwoBankBenchmarkDiscriminator`. Lower theoretical purity (no likelihood; calibration stays percentile-based), but **high practical floor**: it inherits two-bank's 0.832 as a near-guarantee. **Trigger**: OE-density's gain over two-bank is < 5pp.
2. **Low-rank logistic (the previous primary).** `s_logit(z) = w₂ᵀ f(z) + b₂` trained by BCE on `{succ, gt-fail-after-gt}`. Demoted because (i) embeds training prior-odds into the logit, requiring extra calibration; (ii) edge variance is dominated by `N_fail`, the rare class; (iii) is a discriminative special case of OE under specific `p_θ` and loss choices, so OE strictly subsumes it conceptually. **Kept as a §4.G ablation row**, run in the same Phase-2 batch on the same data split.
3. **NCE with constrained `p_data` (candidate C).** Treat fail as noise distribution samples and use NCE to estimate parameters of a constrained `p_succ` (e.g. Gaussian on `f(z)`). Theoretically elegant — fail samples *lower* succ-estimate variance rather than introduce variance of their own. **Trigger**: A and B both plateau, and the failure mode looks like succ-density mis-specification.

## 5. Phased rollout

### Phase 2 — implementation [next, committed]

**Priority A — OE-density (this plan's primary):**
1. Implement `OEDensityScore` under `lpb_v2/models/score_oe_density.py`:
   - Linear `f` (§4.A), full-covariance Gaussian `p_θ` (§4.B), hinge with data-driven margin (§4.C), joint optimization with `λ`, `γ` per §4.D.
   - `.fit(z_succ, z_fail, task_id)` and `.score(z, task_id) -> Tensor`. Save/load state dict.
   - Unit-test on synthetic 2-class Gaussians: recover the planted projection direction; verify `λ=0` reduces to PCA-style succ-only Mahalanobis on a learned subspace.
2. Build `OEDensityBenchmarkDiscriminator` mirroring `TwoBankBenchmarkDiscriminator` plumbing — frozen encoder via `_encode`, three-way disjoint splits (eval / f-train / fail-calib if Youden) by `video_id`, hard-asserts.
3. Train + evaluate on the four AgileX tasks with **10 fail trajs/task for f-train**, disjoint from the 10 used as the two-bank reference and from the 15 eval trajs/task. Pooled with task one-hot per §4.E.
4. Run all §4.G baselines on the same eval split in one batch; emit single JSON for the head-to-head table. Specifically include the `λ=0` row (this is the most important ablation).

**Priority B — Deep one-class metric learning + two-bank (§4.J #1):** run if A's gain over two-bank is < 5pp pooled frame AUROC, or if any task regresses > 3pp under A.
- Same `f` architecture as A, different loss: succ-NN-pull + fail-NN-push hinge.
- Scoring reuses existing two-bank KNN in `f(z)` space — drop-in to `TwoBankBenchmarkDiscriminator` with a learned encoder wrapper.

**Priority C — Low-rank logistic (§4.J #2):** run as a §4.G ablation row in the same Phase-2 batch, **not as a separate research thrust**. Same data split, BCE loss, same `f` rank. Ensures we report a clean head-to-head against the previous primary method.

**Priority D — NCE-with-constrained-`p_data` (§4.J #3):** defer until A and B are benchmarked and the diagnosis (if any) implicates succ-density mis-specification.

### Phase 3 — extensions [contingent on Phase 2 result]
1. **Escalate `p_θ`**: Gaussian → GMM(2..4) → KDE → flow on `f(z)`. Only if A's gain stalls and per-task inspection points at multi-modal success density.
2. **Hybrid score**: `s_total(z) = α · s_OE(z) + (1-α) · s_two_bank(z)`. Justified only if A and two-bank have complementary per-task errors.
3. **Failure-mode discovery on `f(z)`**: cluster gt-fail-after-gt projections, then per-mode density. Requires ≥ 30 fail trajs/task or unsupervised modes that survive cross-validation.
4. **Nonlinear `f`** (2-layer MLP, same `k`) — only after the linear ceiling is mapped via the §4.H k-sweep.

## 6. Implementation TODOs (committed)
1. **`lpb_v2/models/score_oe_density.py` — Priority A (primary).** Linear projection (§4.A) + full-covariance Gaussian density (§4.B) + data-driven hinge (§4.C) + joint loss (§4.D). Pooled with task one-hot per §4.E. `.fit(z_succ, z_fail, task_id)`, `.score(z, task_id) -> Tensor`, save/load state dict. Unit tests on synthetic 2-class Gaussians.
2. **`lpb_v2/benchmark_score.py` — `OEDensityBenchmarkDiscriminator`.** Reuses `_encode` from `LPBV2BenchmarkDiscriminator`. Three-way disjoint splits by `video_id` (eval from `FailureBenchmark`, f-train pool, optional fail-calib pool if Youden). Hard-asserts mirror `TwoBankBenchmarkDiscriminator` (see `PROMPT_two_bank_knn.md` §2).
3. **`lpb_v2/run_score_real_world_benchmark.py` + `scripts/run_score_real_world_benchmark.sh`** — CLI / bash entries modelled on the two-bank versions. Flags: `--ftrain-per-task` (default 10), `--score-k` (default 16), `--lambda` (default 1.0), `--margin-quantile` (default 0.05), `--weight-decay`, `--density {gaussian, gmm2, gmm4}`, `--calib-mode {success_percentile, two_class_youden}`.
4. **`lpb_v2/run_score_ablations.py`** — single script that runs all §4.G comparators on the same eval split and emits one JSON for the head-to-head table. Must include the `λ = 0` ablation explicitly.
5. **`lpb_v2/models/score_logit.py` — Priority C ablation only** (the previous primary, now demoted). Same `f`, BCE loss. Built second so it shares `f` plumbing with §6.1.
6. **`lpb_v2/models/score_metric_two_bank.py` — Priority B (deferred until A is benchmarked).** Metric-learning loss (succ-NN-pull + fail-NN-push); scoring reuses two-bank KNN in `f(z)` space.
7. A short `PROMPT_oe_density.md` companion (parallel to `PROMPT_two_bank_knn.md`) to hand off Phase-2 implementation of §6.1 to a coding agent.

## 7. References
- **Hendrycks, Mazeika, Dietterich.** *Deep Anomaly Detection with Outlier Exposure*, ICLR 2019. Direct precedent for the §2 / §4.A–§4.D design: use an auxiliary small "outlier" set as a margin-style auxiliary loss on an in-distribution density / classifier.
- **Ruff et al.** *Deep One-Class Classification* (Deep SVDD), ICML 2018; and *Deep SAD: Semi-Supervised Anomaly Detection*, ICLR 2020. Theoretical basis for the §4.J #1 candidate B (deep one-class with negative anchors).
- Vovk, Shafer, Gammerman. *Algorithmic Learning in a Random World*, 2005. Split conformal prediction; finite-sample coverage. Used in §4.F default calibration.
- Gretton et al. *A Kernel Two-Sample Test*, JMLR 2012. MMD + permutation test (Phase 1 diagnostic).
- Ledoit, Wolf. *A well-conditioned estimator for large-dimensional covariance matrices*, J. Multivariate Analysis 2004. Shrinkage covariance for the Mahalanobis baseline.
- Lee et al. *A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks*, NeurIPS 2018. Mahalanobis OOD detection on deep features.
- Gutmann, Hyvärinen. *Noise-Contrastive Estimation*, AISTATS 2010. Theoretical basis for §4.J #3 candidate C.
- Goldberger, Roweis, Hinton, Salakhutdinov. *Neighbourhood Components Analysis*, NeurIPS 2004. NCA ablation against logistic.
- Kirichenko, Izmailov, Wilson. *Why Normalizing Flows Fail to Detect OOD Data*, NeurIPS 2020. Cautionary reading for the Phase-3 flow-based extension of `p_θ`.

---

*Living document. Author: working notes, 2026-05-12. Companion to the broader brainstorm at `robosuite/discriminator/IDEAS_failure_supervision.md` and to the two-bank baseline prompt at `PROMPT_two_bank_knn.md`.*
