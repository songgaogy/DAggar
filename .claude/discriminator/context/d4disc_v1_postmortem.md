# D4-Disc v1 — Post-mortem & Fixes

> First end-to-end training run (`checkpoints/d4disc/dynamics/d4dyn_20260422_050952/`)
> collapsed into the γ=0.5 symmetric fixed point that the design acknowledged
> in §3.3 but relied on "training noise" to escape. Benchmark scored at random
> (AUROC ≈ 0.50). This doc records the diagnosis and the fixes landed in the
> package. Source: `.claude/discriminator/d4_disc_design.md`,
> `.claude/discriminator/d4_implementation_plan.md`,
> `.claude/discriminator/context/d4disc_v1_implementation.md` (not yet written
> — see below).

---

## 1. Observed failure

### 1.1 Run artifacts

- Training log: `checkpoints/d4disc/dynamics/d4dyn_20260422_050952/train.log`
  (last 40+ lines — only Phase B visible, shows all 40 bootstrap epochs with
  the same stuck statistics).
- Checkpoints: `d4_dynamics_Bep{0010,0020,0030,0040}.pt`.
- Benchmark JSON: `checkpoints/d4disc/eval/run_20260422_052820/benchmark.json`.

### 1.2 Benchmark result (random)

| metric | pooled | PickPlaceBread | Can | Cereal | Milk |
|---|---|---|---|---|---|
| trajectory AUROC | **0.510** | 0.527 | 0.454 | 0.537 | 0.475 |
| trajectory AUPRC | 0.464 | 0.582 | 0.468 | 0.561 | 0.376 |
| frame AUROC | **0.503** | 0.216 | 0.417 | 0.791 | 0.483 |
| frame F1 | 0.020 | 0 | 0.06 | 0 | 0 |
| detection_delay_median | **−202** | −258 | −173 | −48 | −202 |

- `tpr@fpr=0.05 ≈ 0.05` — essentially random.
- `detection_delay = −200` — "first predicted failure frame" tends to be
  200 frames *before* the ground-truth failure, i.e. spurious early firing.

### 1.3 Training-time symptoms (Bep0040)

From `health_history[-1]`:

| scalar | value | design-expected |
|---|---|---|
| `mean_gamma` | **0.495** | far from 0.5 by epoch 40 |
| γ q05..q95 | **[0.03, 0.49, 0.50, 0.51, 0.96]** | bimodal or wide-tailed |
| `separation_gap = r⁻ − r⁺` | **0.05** | growing through bootstrap |
| `conditional_delta = E‖f(+) − f(−)‖²` | **0.0026** | > 0.01 |
| `alpha_used` | **1.06** | growing with `k^0.75` |

Observables that made the failure mode obvious:
- γ central 50% ∈ [0.49, 0.51] → 90% of the population is glued to the fixed point.
- `r⁺ ≈ 1.44`, `r⁻ ≈ 1.50` → f(+) and f(−) are numerically the same function.
- `conditional_delta ≈ 0.003` after 40 epochs × ~700 steps ≈ 28k SGD steps.
- `alpha_used` barely moves despite `alpha_k = α₀·(1+k)^0.75` scheduling.

---

## 2. Why the method failed

### 2.1 The symmetric fixed point

Design §3.3 writes the filter as
$$\gamma_j = \sigma\!\bigl(\alpha_k\,(-A_\theta(\varphi_j, a_j, \varphi_{j+h}) - \kappa_k)\bigr)$$
with advantage $A_\theta = (r^- - r^+)/(2\sigma^2)$ computed from two critic
branches $f(\cdot \mid c=+)$ and $f(\cdot \mid c=-)$. The design acknowledges
that at init "$f^+ \equiv f^-$ ⇒ $A=0$ ⇒ $\gamma=0.5$", calling this a
"safe uniform prior" and assuming that **training noise** breaks the
degeneracy. That assumption is wrong for our implementation because three
separate mechanisms conspire to preserve the symmetry:

**1. AdaLN-Zero keeps f(−) ≡ f(null) ≡ f(+) at Phase A end.**
Phase A never routes fail_raw samples to c=−, so the c=− modulation heads
receive zero gradient. With zero-init AdaLN gates, the c=− output equals the
unconditional output — which equals the c=+ output that *was* trained, because
the modulation gates for c=+ are also near-zero when cond_embed magnitudes
are small. Net effect: at the start of Phase B, all three conditions produce
essentially identical predictions.

**2. γ=0.5 is a closed-loop attractor under soft routing.**
When γ≈0.5, each fail sample is sampled 50/50 between c=+ and c=−. Both
branches therefore see identical data distributions and Adam's averaged
gradient pulls them in the same direction. f(+) and f(−) stay identical,
A stays 0, γ stays 0.5.

**3. `alpha_cap_rate=1.0` forbids escape.**
`clamped_alpha = alpha_cap_rate / held_out_r_plus`. With
`held_out_r_plus ≈ 1.0` (summed over latent_dim=256), cap ≈ 1.06. Even if
a sample happens to get A=0.05, σ(1·(−0.05))=0.487 — barely off 0.5. The
γ-EMA(α=0.5) then blends toward 0.5 every epoch. No dynamic range → no escape.

### 2.2 Why "noise" can't help

The design's implicit symmetry-breaking assumption is that SGD mini-batch
noise perturbs f(+) and f(−) differently. Empirically this doesn't hold:

- `condition sampling` routes each fail sample to exactly one of {+, −, ∅},
  and the *expected* gradient over a batch with γ=0.5 is identical for both
  branches. The realized variance of the difference is `O(1/√batch_size)`,
  which averages out across ~28k SGD steps.
- When the realized difference is nonzero, the next gate update sees a tiny
  A, produces γ slightly off 0.5, which routes *slightly* asymmetrically in
  the next epoch. But with α capped at 1 and the γ-EMA at 0.5, the
  correction is O(0.01) per epoch — buried under the next noise realization.

The system is a **locally stable fixed point** under our configuration: small
asymmetric perturbations decay. Only a **finite-amplitude** perturbation with
the right sign structure can break symmetry.

### 2.3 Inconsistent per-task frame AUROC is the fingerprint

Per-task frame AUROC ranges from 0.22 to 0.79 across the four eval tasks.
Variation this wide with an essentially-unconditional model means the score
is dominated by an *uncontrolled* signal — whatever residual the
unconditional dynamics predictor happens to have on each task. Had the model
learned a real failure detector, AUROC would be consistently >0.5 across
tasks (even if modest).

---

## 3. Fixes landed in the package

Each fix targets one of the three mechanisms above. They are additive: any
one alone likely insufficient; together they produce a finite-amplitude
asymmetric perturbation with the correct sign and enough training budget
to amplify it into a real separation.

### 3.1 F3-based γ warm-start (breaks mechanism 2)

`robosuite/discriminator/d4disc/filter.py :: warm_start_gamma_from_knn`

At the Phase A→B boundary, seed γ with D3's F3 signal — KNN squared distance
of each fail_raw `z_t` to the pooled clean-positive `z_t` bank, passed
through a sigmoid with auto-β (1/MAD of the bank's self-kNN) and auto-κ
(median of that distance). On-manifold prefix frames get γ≈0, genuinely
off-manifold frames get γ≈1. Written directly to `gamma_buffer`
(`ema_alpha=0`, one-shot replace).

- Unit test: `tests/test_warm_start.py` verifies the off/on-support gap
  exceeds 0.3 on synthetic data.
- Wired into `D4Trainer.run()` right after the `alpha0 = 1/MAD` snapshot at
  end of Phase A, before the first Phase B iteration.

### 3.2 `gate_freeze_epochs` (prevents 3.1 from being wiped)

`D4TrainerConfig.gate_freeze_epochs` (shell default 4, python default 2).

For the first N bootstrap epochs, **skip the `compute_advantage_gate` call**
and use the warm-started γ unchanged. Rationale: at that point f(+)≈f(−)
still holds, so recomputing γ from advantage would immediately reset it to
σ(0)=0.5, blend with the warm-start under γ-EMA=0.5, and drag γ back toward
0.5 in ~1 epoch. Freezing lets the asymmetric routing (predominantly c=−
for warm-started-high-γ samples) train f(−) to predict fail targets, growing
`separation_gap` before the gate takes over.

### 3.3 `alpha_cap_rate: 1.0 → 5.0` (restores α dynamic range)

`D4Schedule.alpha_cap_rate = 5.0`; shell default also 5.0. With
`held_out_r_plus ≈ 0.9` in practice, `cap = 5/0.9 ≈ 5.5` — enough to let α
follow the scheduled `α₀ · (1+k)^0.75` during the first ~30 bootstrap
epochs instead of being flat at 1.

### 3.4 `adaln_init_std: 0.0 → 0.05` (breaks mechanism 1 structurally)

`AdaLNModulation.__init__(..., init_std=0.05)`; shell default 0.05 (python
class default 0.0 for backward compatibility with the identity-at-init unit
test). With `init_std=0.05` the modulation head's linear produces a small
nonzero (shift, scale, gate) for every condition at step 0, so f(+), f(−),
f(∅) are structurally different from the first forward pass. Measured on
the real-scale predictor:

| `adaln_init_std` | conditional_delta at init (sum/dim, 256-D) |
|---|---|
| 0.00 | 0.0 (identity) |
| 0.02 | 0.39 |
| 0.05 | ~2.3 |
| 0.10 | 9.3 |

Compare to the failed run's end-of-training `conditional_delta ≈ 0.0026`
(same metric, scaled by 1/256). At 0.05 the initial differentiation is
already hundreds of times larger than what 40 epochs of training managed
under AdaLN-Zero. This is the most direct fix; it sacrifices the
plan-level "identity at init" invariant, but that invariant was precisely
what prevented f(−) from ever diverging.

Unit tests for AdaLN-Zero identity still pass because they construct blocks
with explicit `init_std=0.0`.

### 3.5 γ quantile logging

`filter.py :: _gamma_quantile_diag` reports q05/q25/q50/q75/q95 of γ over
fail_raw samples. Included in every warm-start and bootstrap health
snapshot (wandb scalars + stdout). Lets the failure mode be diagnosed in
the first 2-3 epochs instead of 40. If `q25 ≈ q75` is near 0.5, you're in
the fixed point — restart with more aggressive knobs.

### 3.6 Defaults summary (after fixes)

| knob | old | new (shell) | python class |
|---|---|---|---|
| `F3_WARM_START` | N/A | 1 | True |
| `GATE_FREEZE_EPOCHS` | N/A | 4 | 2 |
| `ALPHA_CAP_RATE` | 1.0 | 5.0 | 5.0 |
| `ADALN_INIT_STD` | N/A | 0.05 | 0.0 |

---

## 4. What to watch on the next run

Canonical sequence for an "alive" run:

**End of Phase A (epoch 8)**
- `warmstart/mean_gamma` ∈ [0.4, 0.8]
- `warmstart/gamma_q05 < 0.3` and `warmstart/gamma_q95 > 0.7`

**Frozen epochs (bootstrap_k = 0..3)**
- γ quantiles unchanged (`gate_frozen=1`).
- `latent_mse` decreases from the Phase A end value.
- `conditional_delta` grows from `adaln_init_std^2 * O(d_model)` toward
  larger values — if it *doesn't* grow, fail data is too weak or batches
  aren't routing c=−; check `cond_minus` count in training log.

**Gate active (bootstrap_k ≥ 4)**
- `mean_gamma` moves away from 0.5 in the direction the warm-start set
  (not back toward 0.5).
- `separation_gap > 0` and growing.
- `alpha_used` follows `α₀ · (1+k)^0.75` until capped; cap is ~5.5 so it
  should plateau around epoch 15-20 not immediately.

**Collapse signals (any of these → abort and retune)**
- `mean_gamma` returns to [0.45, 0.55] range by k=8.
- `separation_gap` decreases or stays < 0.1 past k=10.
- `conditional_delta` not monotone.

### Escalation ladder if still stuck

1. `ADALN_INIT_STD=0.1` (stronger symmetry break at init).
2. `GATE_FREEZE_EPOCHS=8` (longer training before gate takes over).
3. `ALPHA_CAP_RATE=10.0` with `EMA_ALPHA_GAMMA=0.3` (more cap, less γ
   inertia).
4. Increase capacity (`NUM_LAYERS=6`, `NUM_HEADS=8`) and batch size — with
   d_model=512 × num_layers=4 the model may be under-parameterized for
   learning two distinct dynamics.

---

## 5. Files modified (all under `robosuite/discriminator/d4disc/`)

| file | change |
|---|---|
| `filter.py` | new `warm_start_gamma_from_knn`, `_gamma_quantile_diag`; γ-quantile fields added to `compute_advantage_gate` return |
| `schedule.py` | `alpha_cap_rate` default 1.0 → 5.0 |
| `adaln.py` | `AdaLNModulation(init_std=0.0)` + `AdaLNDecoderBlock(adaln_init_std=0.0)` params |
| `model.py` | `ConditionalDynamicsPredictor(adaln_init_std=0.0)` param; propagated into blocks + final mod; `arch_args` records it |
| `trainer.py` | `D4TrainerConfig.{f3_warm_start, f3_warm_start_k, f3_warm_start_max_clean, gate_freeze_epochs}`; `run()` calls warm-start after Phase A; gate skipped while `k < freeze_epochs`; γ-quantile in health dict; `gate_frozen` flag in history entry |
| `train_d4.py` | `--f3-warm-start/--no-f3-warm-start`, `--f3-warm-start-k`, `--f3-warm-start-max-clean`, `--gate-freeze-epochs`, `--adaln-init-std` CLI flags |
| `scripts/train_d4.sh` | `F3_WARM_START`, `F3_WARM_START_K`, `F3_WARM_START_MAX_CLEAN`, `GATE_FREEZE_EPOCHS`, `ALPHA_CAP_RATE=5.0`, `ADALN_INIT_STD=0.05` env vars |
| `tests/test_warm_start.py` | new — off-support γ > on-support γ on synthetic data |
| `README.md` | "Symmetric fixed-point failure mode" section with the full knob table |

---

## 6. Design vs implementation — what does this say?

The design (`.claude/discriminator/d4_disc_design.md` §3.3, §6) treats γ=0.5
as a benign "uniform prior" because its proof sketch uses a Lipschitz
contraction argument that holds once *any* finite gap between f(+) and f(−)
exists. In practice, opening that gap is not free — AdaLN-Zero makes it
effectively impossible without an explicit symmetry-breaking step.

The fix preserves the design's GPI structure (Sweep I = critic eval with
γ-weighted routing, Sweep II = γ update from advantage). What changed is
the **initialization of the fixed-point iteration**: instead of starting at
(γ=0.5, f(+)=f(−)=f_A) — which is a fixed point — we start at
(γ=F3_warm_start, f(+)=f(−)=f_A + small_perturbation) and freeze the Sweep
II updates for the first few Sweep I iterations. This puts the iteration
into the basin of attraction of the informative fixed point described by
design §6.4, at the cost of biasing γ toward the D3-style KNN signal early.
The asymptotic claim of design §6 (exact recovery of the true off-manifold
indicator) is unchanged — we are adjusting only the initialization, not the
operator.

One honest consequence: **D4 is no longer strictly more principled than
D3**. In particular, the F3 warm-start is D3's filter, so at `k=0` our
model's γ *is* D3's filter. The claim "D4 = D3 at one GPI step" from design
§7.2 becomes literal: without at least a few post-warm-start bootstrap
epochs during which f(−) actually differentiates, D4's γ is just D3's γ.
Whether real bootstrap refinement beats that one-shot D3 signal is the
empirical question the next training run has to answer.
