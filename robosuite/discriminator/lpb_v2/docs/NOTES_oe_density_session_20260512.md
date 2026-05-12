# OE-Density Phase 2 — 2026-05-12 Session Summary

Continuation of `NOTES_oe_density_phase2.md`. Captures the loss-form fix, the
training-data overhaul, three end-to-end runs, and the open analysis of why
frame AUROC plateaus at ~0.69 despite trajectory AUROC reaching 0.97.

Target metric: pooled **frame** AUROC on real-world AgileX, ≥ 0.882
(+5pp over two-bank KNN baseline 0.832).

---

## 1. Run timeline

| Run | Loss form | Data | Frame AUROC | Traj AUROC | Detection delay | Status |
|---|---|---|---|---|---|---|
| `run_20260512_205026` | hinge (dying) | `[first_gt, +60)` only | 0.570 | 0.794 | −127 | loss died ep1 |
| `run_20260512_211335` | anti-density (λ=1, unbounded) | `[first_gt, +60)` | 0.796 | 0.764 | −17 | L_total → −2×10²¹ |
| `run_20260512_212831` | **logistic** | `[first_gt, +60)` | 0.576 | 0.946 | −111 | loss bounded; traj-level shortcut |
| `run_20260512_214039` | **logistic** | **fail=`[first_gt, T)`, succ += fail prefix** | **0.692** | **0.973** | −110 | best so far |

Two-bank KNN baseline (yardstick): frame AUROC **0.832**.

## 2. Fix 1 — loss form: anti-density → logistic

**Diagnosis.** At λ=1 the anti-density loss
`L = −E_s log p + E_f log p = log(p_fail/p_succ)` is mathematically
unbounded for a Gaussian: shifting µ along the discriminative axis sends
`q_s − q_f → −∞`. W1 row-normalisation only blocked one flat direction;
µ stayed unconstrained. Run 211335 diverged to L_total ≈ −2×10²¹.

**Fix.** Logistic discriminative loss on the score `s(z) = −log p_θ(f(z))`
with a learnable scalar threshold `τ`:
```
L_succ = mean softplus( s(z_s) − τ )
L_fail = λ · mean softplus( τ − s(z_f) )
```
Bounded below by 0, saturates as samples are correctly classified.

`τ` is seeded at `median(s(z_succ_train))` in `init_density_from_batch` so
the loss starts in its linear regime. Training-time `τ` is distinct from
the adapter's per-task percentile threshold used at test time.

**Tests.** Added `test_loss_stays_bounded_under_long_training` regression
guard. All 7 unit tests pass.

**Files changed:**
- `models/score_oe_density.py` (new `tau` Parameter, rewrote `loss()` and
  docstring, `init_density_from_batch` seeds `tau`)
- `benchmark_oe_density.py` (logging, calibration_summary)
- `tests/test_score_oe_density.py` (regression guard)

## 3. Fix 2 — training data: paired same-trajectory supervision

**Motivation.** Run 212831 had loss bounded but frame AUROC collapsed to
0.576 — the logistic loss found the *globally optimal* classifier for the
training labels, which is a **trajectory-level shortcut**: every frame of
a success trajectory ≠ every frame of a failure trajectory's tail can be
separated by scene/setup statistics, not failure-vs-completion semantics.

**Fix (user's suggestion).**
- **fail pool**: all frames `[first_gt, T)` of each failure trajectory (10
  per task), not just `[first_gt, first_gt + last_k)`.
- **succ pool**: success-rollout frames **+** the `[0, first_gt)` prefix
  of each failure trajectory.

The prefix is from the *same* failure trajectory as the suffix — paired
same-scene/setup supervision, so the only remaining signal the model can
pick up is failure-vs-completion, not trajectory identity.

**Implementation.** Two new flags on `OEDensityBenchmarkDiscriminator`,
both default True:
- `fail_use_all_after_gt`: use `[first_gt, T)` instead of `[first_gt, +last_k)`.
- `fail_prefix_to_succ`: add `[0, first_gt)` to `pooled_succ_train`.

New helper `_encode_fail_split` returns `(prefix, suffix)` after a single
encode pass per trajectory. Per-task counts logged in
`_calibration_stats[task]["num_succ_from_fail_prefix_*"]`.

**Result.** Run 214039: frame AUROC 0.576 → 0.692 (+11.6pp), trajectory
AUROC 0.946 → 0.973 (+2.7pp), TPR@FPR=0.05 0.60 → 0.79. Detection delay
unchanged at −110.

## 4. Remaining gap analysis

Frame AUROC 0.692 still well below two-bank KNN 0.832 and target 0.882.
The fail-loss form is healthy (bounded, converged). The trajectory-level
shortcut is gone. So what's left?

Three layered causes, in decreasing order of importance:

### 4.1 `first_gt` is a semantic label, not a physical failure moment

Annotators mark `first_gt` when failure is *obvious*. Physical failure
precursors (grasp angle drift, missed contact) appear 50-150 frames
earlier. The model learns precursors, fires at the precursor frame.
Detection delay = `first_gt − model_fire` = +100s = correct physics, wrong
ground truth. Frame AUROC uses the `first_gt` label as ground truth, so
the model's early-but-correct firing is penalised as false positive.

**Evidence.** detection_delay_median = −110, frame_precision = 0.71 (when
model fires, often right), frame_recall = 0.34 (misses most "official"
fail frames).

### 4.2 Multi-modal succ collapsed to unimodal Gaussian

Real succ distribution is multi-modal in latent space (task phases:
approach / grasp / transport / place — each phase has distinct latent
statistics). Our architecture is:
```
z (551-D) → W1 (16-D linear projection) → Gaussian(µ, Σ) → s = −log p
```
A 16-D unimodal Gaussian on a learned linear projection cannot represent
multi-modal succ. The optimisation finds *one* projection direction where
training succ and training fail are separable, and tightens `Σ` to a
small ball around the projected succ mean (empirically `q_succ ≈ 2`
vs k=16 expected).

**Why this matters at test time.** The W1 projection is overfit to the
specific contrast between training succ and 10 training fail trajectories.
Test failure modes not captured by W1 project anywhere in 16-D — including
into the succ ball — and are scored as succ. Hence frame recall 0.34.

This is the failure mode the user diagnosed correctly ("Gaussian doesn't
fit the real distribution"). The precise mechanism is **projection
overfit + unimodal density**, not "Gaussian covers too large area"
(empirically Σ is *tight*, not wide).

### 4.3 softplus learns a smooth boundary

Even with perfect labels and a multi-modal density, softplus's smooth
classifier doesn't match the 0/1 step at `first_gt`. The decision
boundary leaks into the late prefix and early suffix.

## 5. Why two-bank KNN works (single-frame, no temporal context)

KNN score per frame = `d_succ_NN(z) − d_fail_NN(z)`. Single-frame, no
trajectory bookkeeping. Wins on three axes:

1. **Non-parametric → preserves multi-modality.** Each task phase keeps
   its own bank samples; no Gaussian collapse.
2. **Local distance → same-phase comparison.** Test frame compared to its
   actual nearest training neighbours (likely same phase), not a global
   centroid.
3. **No learned projection.** Uses raw 551-D latent, no W1 overfit.

This validates: the bottleneck in OE-density is the (W1 projection +
unimodal Gaussian) representation, not the loss form.

## 6. Path forward — single-frame discriminator

User constraint: stay single-frame, no temporal/trajectory reasoning.

Ranked by expected payoff / cost:

1. **KNN scoring on learned `f(z)` features** (`~100` lines).
   Keep current W1 training, but at test time replace `−log p_Gauss(f(z))`
   with `d_succ_NN(f(z)) − d_fail_NN(f(z))`. Diagnostic: if KNN on `f(z)` >
   KNN on raw latent (i.e. > 0.832 frame AUROC), W1 is a useful feature
   space and Gaussian was the bottleneck. If ≈, W1 itself is destroying
   information.
2. **GMM2 / GMM4 density** (`~50` lines). Code already has
   `_VALID_DENSITIES = ("gaussian", "gmm2", "gmm4")` reserved for Phase 3.
   Cheapest expressivity bump.
3. **`score_k` sweep** (`16 → 32 → 64`). One config change. Determines
   whether projection dimensionality is a bottleneck.
4. **Soft `first_gt` labels.** Ramp the label from 0 → 1 over a window
   around `first_gt` instead of a step. Matches softplus's smooth
   boundary to the ground truth's smoothness.
5. **MLP head + flow density.** Heavy. Overlaps with Phase 3 escalation.
   Only if 1-4 plateau.

Open question for next session: do #1 first (cheapest discriminative
experiment). The result tells us whether to invest in better density
(GMM/flow) or better features (MLP head).

## 7. Run artefacts

- Loss-form fix only:
  `checkpoints/lpb_v2/oe_density_eval/run_20260512_212831_oe_density`
- Loss-form fix + data overhaul (current best):
  `checkpoints/lpb_v2/oe_density_eval/run_20260512_214039_oe_density`
- Diverged second run (anti-density, λ=1):
  `checkpoints/lpb_v2/oe_density_eval/run_20260512_211335_oe_density`
- Broken first run (dying hinge):
  `checkpoints/lpb_v2/oe_density_eval/run_20260512_205026_oe_density`
- Two-bank KNN yardstick:
  `checkpoints/lpb_v2/two_bank_eval/run_20260512_164541_two_bank_diff_a1`

## 8. Files in current state

- `models/score_oe_density.py` — logistic loss, learnable `tau`
- `benchmark_oe_density.py` — fail_use_all_after_gt + fail_prefix_to_succ
  flags wired through `_encode_fail_split`
- `run_oe_density_real_world_benchmark.py` — CLI surface for the two new
  flags (both default True), manifest records them
- `tests/test_score_oe_density.py` — 7 tests, all pass, includes bounded-loss
  regression guard
- `NOTES_oe_density_phase2.md` — original Phase 2 working log, updated
  with §4b/§4c (loss-form revision history)
- `NOTES_oe_density_session_20260512.md` — this file
