# D⁴-Disc: Conditional Collapse Analysis & Mitigation Roadmap

> **Date**: 2026-04-24
> **Scope**: Diagnosis of the pos/neg condition collapse problem in Phase-B
> bootstrap, survey of candidate fixes (InfoGAN, Kingma M2, anchor loss,
> contrastive, structural asymmetry, etc.), and the recommended route going
> forward.
> **Companion docs**:
> - `d4disc_0423.md` — code-aligned architecture/training/inference reference
>   (updated 2026-04-24 with KNN-advantage-gate as Phase-B default)
> - `d4disc_0424_debug_lpb_degraded.md` — motion-confound root cause behind the
>   `rel` score mode
>
> This doc does *not* prescribe code changes yet. It is the analysis/decision
> layer feeding the next implementation PR.

---

## Table of Contents

1. [Problem diagnosis](#1-problem-diagnosis)
2. [Why the existing three defenses do not solve it](#2-why-the-existing-three-defenses-do-not-solve-it)
3. [Candidate solutions surveyed](#3-candidate-solutions-surveyed)
   - 3.1 [InfoGAN-style mutual information maximization](#31-infogan-style-mutual-information-maximization)
   - 3.2 [Kingma M2 semi-supervised VAE](#32-kingma-m2-semi-supervised-vae)
   - 3.3 [Anchor loss (recommended minimal fix)](#33-anchor-loss-recommended-minimal-fix)
   - 3.4 [Contrastive push on fail data](#34-contrastive-push-on-fail-data)
   - 3.5 [Structural asymmetric head](#35-structural-asymmetric-head)
   - 3.6 [Third-party gate (gate decoupled from predictor)](#36-third-party-gate-gate-decoupled-from-predictor)
   - 3.7 [Extended gate-freeze schedule](#37-extended-gate-freeze-schedule)
4. [Framing correction: KNN is a metric, not a training path](#4-framing-correction-knn-is-a-metric-not-a-training-path)
5. [`cfg_knn`: the missing inference mode](#5-cfg_knn-the-missing-inference-mode)
6. [Recommended path & ablation matrix](#6-recommended-path--ablation-matrix)
7. [Open decision points](#7-open-decision-points)

---

## 1. Problem diagnosis

### 1.1 User-supplied framing (correct, but incomplete)

The MSE loss does not require `c` to carry information. In the Phase-B
bootstrap loop the network simultaneously plays two games — routing (updating
γ) and regression (MSE against `z_target`) — and **`f_+ ≡ f_-` is a stable
Nash fixed point** under the current objective.

### 1.2 Deeper diagnosis — label-switching symmetry in (θ, γ) space

Training objective (ignoring the proprio term, scalar weights elided):

```
L(θ; γ) = E_{j∈D_pos} ||f_θ(z_j,s_j,a_j|+) - z*_j||²
        + E_{j∈D_fail} [ (1-γ_j) r_+(j) + γ_j r_-(j) ]
```

Gate update (Phase-B default, `advantage_mode="knn"`, §5.1):

```
r_±(j)     = min_{b∈B_expert} ||f_θ(…|±) - b||²     # KNN to expert next-latent bank
A(j)       = (r_- - r_+) / (2σ²)
γ_new(j)   = σ( α_k · (-A - κ_k) )
```

**The stable fixed point.** If `f_θ(·|+) ≡ f_θ(·|-)`:

- `r_+ = r_- ⇒ A = 0 ⇒ γ = 0.5` for every fail transition.
- With `γ ≡ 0.5`, the fail-loss reduces to `½(r_+ + r_-) = r_+`, i.e. the
  conditional branch contributes **no gradient that depends on c**.
- Equivalently: the M-step has a degenerate optimum where `c` is ignored; the
  E-step, fed with that optimum, returns `γ = 0.5` uniformly.

This is a **self-consistent fixed point** — not a saddle. AdaLN-Zero + the
zero-init residual head place the initial `θ` exactly at its center, so the
system *starts inside the basin of attraction*.

### 1.3 The symmetry is structural, not just loss-shaped

The label-switching transform `(+, -) ↔ (-, +)` together with `γ → 1-γ` leaves
the loss invariant. Nothing in the current machinery breaks this symmetry from
the inside — γ's initial value from KNN warm-start is the *only* external
asymmetry, and it is erased as soon as the M step pulls `f_-` back toward
`f_+` (which is always available as a low-loss region because `f_+` is already
a reasonable dynamics predictor).

---

## 2. Why the existing three defenses do not solve it

| Defense | Mechanism | Why it fails |
|---|---|---|
| **KNN warm-start** (`f3_warm_start=True`) | Initializes γ_j from KNN rank/sigmoid in encoder latent space | Changes γ, not θ. After a few M-step epochs, `f_-` drifts toward `f_+`, then gate recomputes `A≈0` and γ relaxes to 0.5. |
| **Gate freeze** (`gate_freeze_epochs>0`) | Pauses E-step for first k bootstrap epochs | Only delays the collapse. Once freeze ends, if θ has already landed near the symmetric optimum, the gate has no signal to push it out. |
| **CollapseDetector + EMA rollback** (§7) | Detects γ saturation / `Δ_cond<1e-4` / critic divergence, reverts to EMA weights and halves `alpha_0` | EMA of a collapsing trajectory is itself a collapsed state. Halving α slows the gate further. This is a guardrail that *reports* collapse, not one that *prevents* it. |

All three act on γ or on the *speed* of dynamics. None changes the fact that
the symmetric θ is a stable attractor of the joint (θ, γ) coupled system.

---

## 3. Candidate solutions surveyed

### 3.1 InfoGAN-style mutual information maximization

Add auxiliary classifier `Q_ψ(c | pred_latent)` and a term

```
L_MI = -E_{c, j} [ log Q_ψ(c | f_θ(z_j,s_j,a_j | c)) ]
```

giving `I(c; ẑ) ≥ E[log Q_ψ] + H(c)`.

- ✅ Breaks `f_+ ≡ f_-` — a collapsed θ makes Q uninformative, contributing
  non-zero gradient pushing the branches apart.
- ✅ Cheap: a single `Linear(latent_dim, 2)` head.
- ❌ **No semantic anchoring**: MI is unsupervised; the axis `Q_ψ` discovers
  need not correspond to success/fail dynamics. CFG at inference depends on
  that alignment being exact.
- ❌ **Shortcut risk**: `Q_ψ` can exploit a low-energy orthogonal subspace of
  `pred_latent` (e.g. dimensions unconstrained by MSE) to encode `c`,
  satisfying the MI bound without making the main branches differ along the
  ID/OOD axis.

**Verdict**: not sufficient on its own; only useful as a cherry on top of a
scheme that already anchors `+` / `-` semantically.

### 3.2 Kingma M2 semi-supervised VAE

Treat `c` as partially-observed latent. The generative model is
`p_θ(z* | x, c) p(c)` (Gaussian likelihood × Bernoulli prior) and the inference
model is a **separately parameterized** classifier `q_φ(c | z_t, s_t)`.

**Key observation**: the current D⁴ code is *already* an M2-style EM loop, but
with `q_φ` tied to the likelihood (γ is derived from `f_θ`'s residuals). That
tie is exactly what creates the self-consistent symmetric fixed point. Kingma's
M2 requires `q_φ` to be *independent* and **anchored by the supervised term on
clean positives**:

```
L_sup = E_{j∈D_pos} [ -log q_φ(+ | z_j, s_j) ]       # anchors q_φ(+) to clean pos
L_unsup(fail) = E_{c~q_φ}[-log p_θ(z* | x, c)] + KL(q_φ || prior(c|x))
```

- ✅ Theoretically clean. `q_φ` cannot collapse to 0.5 because clean-pos
  supervision pins it down.
- ✅ γ from `q_φ`, M-step uses `c ~ q_φ`, E-step never uses the predictor.
- ❌ Requires a new network + new loss term + new checkpoint field.
- ⚠️ *Meta-risk*: if `q_φ(c|x)` linearly separates clean/fail, the entire
  d4 pipeline's transition-level value proposition is in question. The
  justification remains: `q_φ` must be imperfect at the transition level
  (some early fail-rollout steps still look like success), and `f_θ` refines
  that soft label into a proper dynamics-aware score.

**Verdict**: most principled long-term solution. Keep in reserve if the
cheaper mitigations don't pan out.

### 3.3 Anchor loss (recommended minimal fix)

Inject a semantic prior that "the `-` branch represents off-manifold
(failure) dynamics" directly into the loss, on **clean positives**:

```
L_anchor = λ_anc · E_{j∈D_pos} [ max( 0, m − min_{b∈B_expert} ||f_θ(z_j,s_j,a_j|-) - b||² ) ]
```

i.e. for clean positives the `-` branch must produce next-latents *far* from
the expert bank, with hinge margin `m`.

**Properties**:

- ✅ Directly breaks `f_+ ≡ f_-` — at the symmetric fixed point clean-pos
  `r_-` is small, the hinge activates, gradient pushes branches apart in a
  **specified direction** (`-` away from bank).
- ✅ No new network. Bank already built once per Phase-B start
  (`filter.py::build_expert_target_bank`), encoder frozen.
- ✅ **Metric-consistent with the Phase-B KNN advantage gate** (§5.1): both E-
  step (gate) and the new M-step term measure distance in encoder latent space
  to `B_expert`.
- ✅ **Semantically aligned with CFG** inference: `f_ω = (1+ω)f_+ - ωf_-`
  wants `+` ≈ expert-manifold, `-` ≈ off-manifold. Anchor loss encodes
  exactly that prior.
- ⚠️ Hyperparameter `m` needs care. Good default: median of `d²` from
  `warm_start_gamma_from_knn` diagnostics (already computed).
- ⚠️ Overly-strong `λ_anc` could pollute `+` branch via shared transformer
  params. Start small (0.1), monitor `held_out_r_plus`.

**Verdict**: **lowest cost / highest ratio**. First thing to implement.

### 3.4 Contrastive push on fail data

```
L_contrast = -λ · E_{j∈D_fail} [ tanh( ||f_+(j) - f_-(j)||² / τ ) ]
```

Tanh caps divergence to avoid blow-up. Weaker than anchor loss: pushes
branches apart but does not specify a direction, so the *identity* of which
branch corresponds to success vs fail is left to the (weak) γ signal.

**Verdict**: fallback if anchor loss has too much impact on `+` branch
quality. Strictly weaker than anchor as a standalone fix.

### 3.5 Structural asymmetric head

Replace the unified residual head with:

```
pred_latent = z_t + Δ_shared(tokens) + sign(c) · Δ_specific(tokens)
```

`Δ_specific` is a zero-init independent MLP. `f_+ - f_-` is deterministically
`2 Δ_specific(tokens)` — non-zero the moment `Δ_specific` gets any gradient.

- ✅ Deterministic symmetry break (vs `adaln_init_std>0` which is stochastic).
- ❌ Does not guide `+`/`-` toward a specific semantic axis; still needs an
  anchor or γ signal to pin orientation.
- ❌ Invasive model change; would require ckpt migration.

**Verdict**: cleaner than raising `adaln_init_std`, but redundant if anchor
loss is in place. Skip unless future experiments need it.

### 3.6 Third-party gate (gate decoupled from predictor)

Replace `r_±` in `compute_advantage_gate` with a static KNN distance from a
frozen feature extractor (e.g. encoder projection or BYOL projection head):

```
gamma_new = σ( α · ( -KNN_external(j) - κ ) )
```

- ✅ Gate decoupled from `θ` → never self-consistently collapses.
- ❌ Loses bootstrap's dynamic refinement; degenerates d4 toward a dressed-up
  d3 + KNN warm-start. Defeats the purpose of a conditional dynamics critic.

**Verdict**: not recommended unless we decide to scrap Phase-B entirely.

### 3.7 Extended gate-freeze schedule

Keep the existing freeze mechanism but extend it: during freeze, keep γ from
warm-start *and* force training to actually use those γ values (current code
already does this). Then release gate only after `conditional_delta` exceeds
a threshold.

- ✅ Zero-cost schedule tweak.
- ⚠️ Insufficient alone — if θ reaches the symmetric optimum during freeze
  (likely, since M step has no incentive to diverge the branches),
  releasing gate won't help.

**Verdict**: useful companion to anchor loss, not a standalone fix.

---

## 4. Framing correction: KNN is a metric, not a training path

Earlier analysis in this conversation conflated two orthogonal axes:

- **Training recipe**: Phase-A only vs Phase-A+B.
- **Scoring mode**: `abs` / `rel` / `knn`.

The three scoring modes are all applied to the *same* trained predictor. The
`knn` mode specifically uses `predictor.obs_proj` / `proprio_proj` /
`action_proj` (input linear projections) as a feature extractor and **bypasses
the conditional decoder entirely**. Those input projections receive gradients
in both Phase A and Phase B. Therefore:

- "Phase-A only + knn = 0.960" is **not** evidence against investing in
  Phase B. It is a baseline indicating that input projections alone suffice
  for KNN-on-features.
- A healthy Phase-B must either (a) improve those same input projections
  (→ knn score goes up) or (b) make the conditional decoder useful
  (→ CFG modes light up). A *collapsed* Phase B may do neither, or even
  pollute the input projections with uninformative gradients.

**Corollary**: the decisive comparison is not "fix Phase-B vs trust knn". It
is **"does fixing Phase-B bring measurable gains under any scoring mode,
KNN included?"**

---

## 5. `cfg_knn`: the missing inference mode

CFG (the `(1+ω)f_+ - ωf_-` combination on decoder outputs) and KNN (using an
expert next-latent bank as the anchor) are **orthogonal design axes**. The
current code exposes three `score_mode`s but misses one natural cell of the
grid:

| Anchor \ Combine | `+`-only | CFG (`f_ω`) |
|---|---|---|
| **Single point `z_target`** | abs / rel with ω=0 | **abs / rel (current)** |
| **Expert bank `B_expert`** | knn (current, also via input-proj features) | **`cfg_knn` — currently missing** |

**Definition**:

```
f_ω(t)       = (1+ω) f_+(z_t, s_t, a_{t:t+h}) - ω f_-(…)        # in encoder latent space
λ_t_cfg_knn  = min_{b∈B_expert} || f_ω(t) - b ||² / (2σ²)
```

Bank `B_expert` already exists from the Phase-B advantage gate
(`filter.py::build_expert_target_bank`) — encoder frozen, so it is valid at
inference time.

**Why this is the natural d4 inference mode**:

- **Train/eval metric alignment**: the Phase-B KNN advantage gate uses
  `min_b ||f_± - b||²` during training. `cfg_knn` uses
  `min_b ||f_ω - b||²` at eval. Both measure distance to the expert
  manifold — same anchor, same distance, with CFG composing on top.
- **Removes the single-target motion confound** (the reason `rel` exists)
  *by construction*, not by post-hoc subtraction.
- **Exercises the conditional decoder** (unlike the current `knn`) so Phase-B
  improvements actually show up in the score.
- **Composition, not new infra**: trivial extension of `D4Detector`.

**Degenerate cases (useful sanity checks)**:

- Phase-A only + `cfg_knn`: `f_-` was never trained (AdaLN-Zero ≈ identity on
  both conditions), so `f_ω ≈ f_+` regardless of ω. Expected to land very
  close to the current `knn` baseline (ω knob inert).
- `ω = 0` + `cfg_knn`: reduces to KNN distance of `f_+` against bank — a pure
  positive-branch density score.
- If `f_+ ≡ f_-` (collapse): `f_ω ≡ f_+`, ω inert. `cfg_knn` still
  inherits the collapse problem — **anchor loss is still required** to realize
  any CFG gain. This is the critical interaction.

---

## 6. Recommended path & ablation matrix

### 6.1 Ranking (cost/reward)

| Solution | Impl cost | Theoretical strength | Risk | Rec |
|---|---|---|---|---|
| **Anchor loss** (3.3) | Very low | Medium (semantic hard-anchor of `-`) | Low | ★★★★★ |
| **Extended freeze** (3.7) | Very low | Low–medium | Low | ★★★★ |
| **`cfg_knn` inference mode** (§5) | Very low | Critical for realizing Phase-B gains | None | ★★★★★ |
| **Structural asymm head** (3.5) | Low | Medium | Medium | ★★ (redundant w/ anchor) |
| **Kingma M2** (3.2) | Medium | Highest | Medium | ★★★★ (reserve) |
| **InfoGAN-MI** (3.1) | Medium | Weak (no anchor) | High (shortcut) | ★ |
| **Third-party gate** (3.6) | Low | Degenerates to d3 | — | ✗ |

### 6.2 Ablation matrix (definitive experiment plan)

| # | Training | Score mode | Purpose |
|---|---|---|---|
| (a) | Phase-A only | `knn` (current) | Baseline ≈ 0.960 pooled |
| (b) | Phase-A+B (current) | `knn` (current) | Is the collapse already polluting input projections? |
| (c) | Phase-A+B (current) | **`cfg_knn`**, ω∈{0.5, 1, 1.5} | How crippled is the conditional decoder right now? |
| (d) | Phase-A+B + **anchor loss** | `knn` (current) | Does fixing collapse spill over into projection quality? |
| (e) | Phase-A+B + **anchor loss** | **`cfg_knn`**, ω>0 | **The definitive test of d4's design premise** |

**Success criteria**:

- (e) > (a): d4's conditional + bootstrap + CFG design is empirically
  validated for the first time.
- (d) > (a): anchor-loss fix brings measurable projection-level gains even
  without invoking CFG — sufficient to merge anchor loss on its own.
- (e) ≤ (a) despite (d) > (a): CFG gives no additional value beyond the
  feature-level improvement; d4 collapses (conceptually) to "LPB with a
  better feature extractor". Decision point for future roadmap.

### 6.3 Implementation order (minimal diff)

1. **Sanity check on current ckpt (half a day, inference-only)**: add
   `score_mode="cfg_knn"` to `D4Detector` + benchmark wrapper. Run on the
   existing Phase-A-only checkpoint. Expect result ≈ (a) since ω is inert
   there. Confirms plumbing works.
2. **Anchor loss (1 day, training)**: add `anchor_weight`,
   `anchor_margin`, `anchor_warmup_epochs`, `anchor_on_fail` to
   `D4TrainerConfig`; bank reuse from Phase-B init; one extra `cond=-`
   forward per batch on clean positives. Log `anchor_hinge_loss` in
   `D4Health`.
3. **Run (b)(c)(d)(e)** and update this doc with numbers.

---

## 7. Open decision points

1. **Name for the new mode**: `cfg_knn` / `knn_cfg` / `cfg_bank` / or fold
   into a unified `cfg` mode that reduces to `knn` at ω=0. Current preference:
   `cfg_knn` as a new enum (non-invasive, keeps `abs`/`rel`/`knn` backward
   compatible).
2. **Anchor margin `m`**: auto (bank `d²` median from warm-start diag) vs
   fixed constant. Preference: auto.
3. **Phase-A `-`-branch warm-up**: Phase A currently trains only `c=+`. The
   `-` branch is cold-started at Phase-B epoch 0, which makes the anchor hinge
   activate from random-ish outputs. Option: add a low-weight `-`-branch
   forward in the last 1–2 epochs of Phase A (anchor-only, no γ-weighted MSE)
   to warm it up. Simpler alternative: keep anchor `anchor_warmup_epochs>0`
   so Phase B's first epochs run without the hinge active.
4. **Meta-question**: if (e) ≤ (a), are we willing to retire Phase B? This
   needs to be pre-committed before running the ablation to avoid
   post-hoc rationalization.
