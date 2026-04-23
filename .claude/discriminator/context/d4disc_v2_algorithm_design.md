# D4-Disc v2 — Algorithm Design Updates

> Context: v1 post-mortem
> (`d4disc_v1_postmortem.md`) landed AdaLN-init + α cap + F3 warm-start +
> gate freeze. Re-running still failed: γ collapsed toward 0.5, `conditional_delta`
> decayed, `branch_starvation` fired at bootstrap k≈5–8. This doc captures
> the **algorithm-side** diagnosis and the fixes that were landed
> (F1/F2/F4 + skip-bootstrap). For the benchmark/ckpt bugs found after
> these fixes, see `d4disc_v2_bench_and_ckpt_bugs.md`.

---

## 1. v1.5 failure (run `d4dyn_20260422_054915`) — what went wrong beyond v1

### 1.1 Evidence

v1 fixes were loaded; new symptoms:

| epoch | k | gate_frozen | mean_γ | γ q05..q95 | sep_gap | cond_δ |
|---|---|---|---|---|---|---|
| warm-start (end of Phase A) | — | — | **0.999** | 0.999..0.999 (ALL) | — | — |
| B009 | 0 | ✓ | 0.999 | 0.999..0.999 | 0 | 0.00200 |
| B010 | 1 | ✓ | 0.999 | 0.999..0.999 | 0 | 0.00054 |
| B014 | 5 | ✗ | 0.625 | 0.46..0.80 | 0.003 | 0.00015 |
| B015 | 6 | ✗ | 0.563 | 0.30..0.83 | 0.0004 | 0.00027 |
| B016 | 7 | ✗ | 0.533 | 0.18..0.89 | −0.0038 | 0.00043 |
| B017 | 8 | ✗ | 0.518 | 0.10..0.94 | **−0.0073** | **branch_starvation** |

### 1.2 Three compounding root causes

**(A) F3 warm-start saturated.** `compute_f3_weights` auto-calibrates
`β = 1/MAD(clean self-kNN)` and `κ = median(clean self-kNN)`. On this
encoder: clean self-kNN `κ ≈ 0.557`, while `fail→clean` d² ranges
roughly `[16, 35, 184]` (q05/q50/q95) — **order of magnitude larger**.
Sigmoid saturates at 0.999 on *every single* fail_raw sample. γ
distribution has zero spread → no informative prior, equivalent to
γ≡1. The intended prefix-vs-catastrophic ranking is destroyed by the
D3-calibrated sigmoid.

**(B) Phase A contaminated f(+) with fail_raw.** Original routing
(v1): fail_raw went to `c=+` or `c=∅` during Phase A. f(+) learned an
*unconditional* dynamics. When Phase B routed fail_raw to `c=-`, f(-)
targeted frames that f(+) had already fit → f(-) reproduces f(+)'s
mapping → `conditional_delta` collapses instead of growing. Deeper
than v1's AdaLN-Zero argument: even with init asymmetry, if both
branches are asked to fit the same function on the same data, they
converge.

**(C) `branch_starvation` detector has an incorrect sign convention.**
`separation_gap = r⁻ − r⁺` measured on fail_raw. If f(-) specializes
correctly, fail samples with high γ have `r⁻ < r⁺` (c=- fits them
better) → mean `separation_gap < 0`. The detector treats this as
"starvation" and halves α₀, pushing the system back toward the
symmetric fixed point it was escaping. v1 post-mortem §4 literally says
"`separation_gap > 0 and growing` → alive signal", which contradicts
what the critic equation implies.

### 1.3 How these compound

- (A) nullifies the warm-start → Phase B starts at uniform γ≈1 (not a
  meaningful prior).
- (B) means even with perfect γ routing, f(+) and f(-) have nothing
  to differentiate on.
- Gate unfreeze sees `|A| ≈ 0` → `γ_gate ≈ σ(0) = 0.5` → EMA drags
  γ back to midpoint within a few epochs.
- (C) fires false-positive collapse and halves α₀, making the already
  too-small `α · |A|` even smaller.

---

## 2. Fixes landed in the code (F1/F2/F4 + skip-bootstrap)

### F1 — rank-based warm-start (breaks root cause A)

`robosuite/discriminator/d4disc/filter.py::warm_start_gamma_from_knn`
gains a `mode` parameter, default `"rank"`.

**Rank mode** (default): `γ_i = (rank_i + 0.5) / N` where rank is over
`fail→clean` kNN d². Produces uniform γ ~ U(0, 1) regardless of scale.
Unaffected by the clean-vs-fail distance-scale mismatch that saturated
sigmoid.

**Sigmoid mode** (kept as option, not the old D3 calibration): auto
`β = 1 / MAD(d²_fail)` and `κ = median(d²_fail)` calibrated from
the **fail→clean distribution itself**, not clean-self. This prevents
saturation but still applies a nonlinearity — use if rank's uniformity
is too aggressive.

Verified on toy data in `tests/test_warm_start.py` (both modes pass the
off-support > on-support gap > 0.3 assertion).

CLI: `--warm-start-mode {rank, sigmoid}`, default `rank`.
Shell: `WARM_START_MODE=rank` default.

### F2 — Phase A clean-only (breaks root cause B)

`trainer.py`:
- `_split_clean_positives` now returns `(train_all, train_clean_only, held_out)`.
- `_train_loader_phase_a` built over clean-only indices.
- `_train_one_epoch(phase)` dispatches loader by phase: A uses
  clean-only loader; B uses the full clean+fail loader.
- `_sample_condition(phase="A")` short-circuits: every sample → `c=+`.
  No `p_cond_drop`, no `c=null` routing. The null branch was vestigial
  (never consumed at inference — detector reads `r⁺`/`r⁻`, never `f(∅)`)
  and was wasting ~50% of Phase A batch slots to satisfy CFG conventions
  we don't use.

Result: Phase A is now a pure supervised regression of f(+) on clean
dynamics. `f(+)` at end of Phase A represents **success-conditional
dynamics** (design intent); Phase B can meaningfully route fail_raw to
`c=-` for f(-) to specialize.

Training log signature: `c+=256 c-=0 cN=0` across Phase A (was
`c+=128 cN=128` with F2 via null routing in an earlier iteration; the
clean-only loader is simpler and cleaner).

### F4 — warm-start diagnostic logging

`warm_start_gamma_from_knn` now returns in the diag dict:

- `d2_mean`, `d2_q05`, `d2_q50`, `d2_q95` — raw `fail→clean` d² stats.
- `raw_gamma_q05`, `raw_gamma_q50`, `raw_gamma_q95` — γ distribution
  BEFORE the `[1e-3, 1-1e-3]` clamp.
- `raw_frac_saturated_hi`, `raw_frac_saturated_lo` — fraction of
  samples that hit the clamp boundaries.

Makes saturation detectable in one line of stdout instead of requiring
a post-mortem.

### Skip-bootstrap mode — safe-fallback detector

`D4TrainerConfig.skip_bootstrap: bool = False`. When `True`:
- Run Phase A normally (clean-only f(+) training).
- Run F3 warm-start to populate `gamma_buffer`.
- Save ckpt.
- **Return before entering the Phase B loop.**

At benchmark time, use `OMEGA=0` so the detector score is
`‖f(+) − z_{t+h}‖²` — no reliance on f(-), which is at init in
skip-bootstrap mode. This is the **"pos-branch-only learned-dynamics
residual"** baseline. If D4's bootstrap cannot extract additional
signal beyond the warm-start (see §3 for why this is plausible on our
data), skip-bootstrap is a lossless degradation to something the user
can still ship.

Shell: `SKIP_BOOTSTRAP=0/1`. CLI: `--skip-bootstrap`.

### Detector sign fix — NOT landed

The `branch_starvation` detector (root cause C) is unchanged. It still
fires on `separation_gap < 0`. Reason: with F1+F2+skip-bootstrap
landed and the benchmark/ckpt bugs still being investigated, we did
not want to debug the detector simultaneously. See `d4disc_v2_bench_and_ckpt_bugs.md`
for why the pipeline health is more urgent. **When resuming Phase B
training, this detector's sign logic must be revisited** — current
behavior actively breaks correct specialization.

---

## 3. Post-F1/F2 run observation (run `d4dyn_20260422_061532`) — still failing

Same benchmark numbers as before the algorithm fixes landed (~random).
Symptoms:

| phase | mean_γ | γ distribution | sep_gap | cond_δ |
|---|---|---|---|---|
| warm-start (rank) | 0.5 | uniform q05=0.05, q95=0.95 ✓ F1 works | — | — |
| B009 (k=0, frozen) | 0.5 | uniform | 0 | 0.00243 |
| B013 (k=4, gate on) | 0.508 | q05=0.27, q95=0.83 (squeezed toward 0.5) | **−0.021** | 0.00016 |
| B014 (k=5) | 0.509 | similar | **−0.017** | 0.00025 (branch_starvation) |

**F1 worked** (γ is now uniform U(0.05, 0.95), not saturated). **F2
worked** (`cond_null=0` in Phase A).

**But `conditional_delta` still collapses** (0.00243 at B009 → 0.00014
by k=3, before gate unfreezes). And gate activation at k=4 immediately
compresses γ toward 0.5.

### 3.1 The deeper theoretical issue

On this encoder (`flow_multi` → 256-D latent), the state `z_t` is close
to Markov — it captures the full physical state. Dynamics are
approximately `z_{t+1} = f(z_t, a_t)` regardless of "this trajectory
is heading toward success or failure". The "success vs failure" label
affects future *action distributions* (via the policy), not the
dynamics operator itself.

Consequence: f(+) trained on clean (z, a) data and f(-) trained on fail
(z, a) data, both minimizing `‖f_c(z,a) − z'‖²`, converge to the **same
function on overlapping support**. They differ only where the two
distributions don't overlap (extrapolation region). The
`conditional_delta` probe on a train-distribution batch samples from
the overlapping region → delta stays small.

B013 diagnostic: `mean_r_plus = 1.76`, `mean_r_minus = 1.74` on
fail_raw. Both branches agree to 1%. `held_out_r_plus ≈ 1.14` on clean.
The signal that separates clean from fail is **"f(+) is worse on fail
than on clean"** (1.76 vs 1.14 = 54% relative) — i.e. the
**distribution shift**, *not* a branch disagreement. This IS
D3/F3's signal (OOD distance), re-expressed in dynamics-residual form.
It is NOT the GPI signal D4 was supposed to recover on top of D3.

### 3.2 Implication

**D4's critic-bootstrap cannot extract additional signal beyond the
warm-start on Markov dynamics with full-state observability.** The
design assumption of §7 in v1 post-mortem
(`D4 = D3 at one GPI step, subsequently refined by training`) breaks
down because the refinement signal is near-zero.

This is NOT a bug — it's a correct theoretical observation about when
GPI-style dynamics bootstrapping helps. It would help more on:

- Partial observability (z is not Markov; c=+ and c=- branches carry
  different *beliefs* about hidden state).
- Noisy / stochastic dynamics (the conditional mean of z' given (z, a)
  actually differs between policy regimes).
- Non-overlapping support (fail trajectories visit states success
  trajectories never do).

Our robot-manipulation rollouts have full state observability + nearly
deterministic simulated dynamics → D4's bootstrap has nothing new to
learn.

**The KNN-based warm-start signal IS the useful signal, and Phase B is
at best a no-op on this data.**

---

## 4. Current recommended detector

Given §3, the practical recommendation:

1. **Skip Phase B** (`SKIP_BOOTSTRAP=1`). Train only f(+) on clean
   dynamics in Phase A. Populate γ via rank warm-start.
2. **Benchmark with `OMEGA=0`**. Score = `r⁺ = ‖f⁺(z,a) − z_{t+h}‖²`.
   This is a "learned clean-dynamics residual" OOD score — a
   model-based D3.
3. Optionally use γ (warm-start KNN ranking) as a separate per-frame
   feature — this is literally D3's F3 filter, available for free from
   the trained dataset.

This is a **D3 upgrade**, not a D4 implementation. But given §3's
theoretical finding and the training pipeline issues documented in
`d4disc_v2_bench_and_ckpt_bugs.md`, it's the honest baseline we can
actually ship.

### 4.1 When to revisit full D4

Full D4 (Phase B bootstrap) only makes sense if the encoder becomes
non-Markov (e.g. if we switch to a partial-observability encoder, or
introduce stochasticity). Don't chase it on the current
`flow_multi` + full-state proprio setup.

---

## 5. Config defaults after this round

| knob | v1 default | current default | location |
|---|---|---|---|
| `F3_WARM_START` | 1 | 1 | shell |
| `GATE_FREEZE_EPOCHS` | 4 | 4 | shell |
| `ALPHA_CAP_RATE` | 5.0 | 5.0 | shell |
| `ADALN_INIT_STD` | 0.05 | 0.05 | shell |
| `WARM_START_MODE` | — | `rank` (NEW) | shell |
| `SKIP_BOOTSTRAP` | — | `0` (NEW; set 1 for KNN baseline) | shell |
| `P_COND_DROP` | 0.1 | 0.1 (but ignored in Phase A after F2) | shell |
| `WARM_UP_EPOCHS` | 8 | 20 (user bumped) | shell |

---

## 6. Files modified in v2 (algorithm-side only)

| file | change |
|---|---|
| `filter.py` | `warm_start_gamma_from_knn(mode=...)`, diagnostics `d2_*` / `raw_gamma_*` / `raw_frac_saturated_*`. Removed dependency on `compute_f3_weights` for the warm-start path (kept as sigmoid fallback with fail-self calibration). |
| `trainer.py` | `D4TrainerConfig.{warm_start_mode, skip_bootstrap}`. `_split_clean_positives` returns 3-tuple; builds `_train_loader_phase_a` clean-only. `_sample_condition(phase=A)` short-circuits to `c=+`. `_train_one_epoch` dispatches loader by phase. `run()` early-returns on `skip_bootstrap`. |
| `train_d4.py` | `--warm-start-mode`, `--skip-bootstrap` CLI flags. |
| `scripts/train_d4.sh` | `WARM_START_MODE`, `SKIP_BOOTSTRAP` env vars. |
| `tests/test_warm_start.py` | Split into rank-mode and sigmoid-mode cases; both pass. |

Not modified: `monitor.py` (detector sign bug remains — see §2 last
paragraph), `adaln.py`, `model.py`, `schedule.py`.

---

## 7. Open algorithm-side questions for next iteration

1. Should the detector's `branch_starvation` rule be sign-flipped, or
   replaced with a stagnation rule (|sep_gap| < ε ∧ mean_γ ∈ [0.45, 0.55]
   for N epochs)?
2. Should `warm_start_gamma_from_knn` also support `mode="topk_quantile"`
   (top-k kNN ranks instead of 1-NN) for robustness?
3. Is the Markov-encoder diagnosis (§3.1) worth testing empirically by
   comparing `f(+)` residual on fail vs clean (expected to differ — D3
   signal) against `(r⁻ − r⁺)` on fail (expected to be tiny — D4 signal)?
   Could be done with a single pass over a trained ckpt, no retraining.
4. Would a **state-diff target** (`target = z_{t+1} − z_t`, then
   `pred_latent` predicts the delta) change the Markov argument? It's
   still deterministic, but the scale of the target shrinks ~100×
   (d²-of-delta is ~0.01 vs 1.0 for absolute latents) which might
   make gradient signal on fail-specific transitions relatively larger.
