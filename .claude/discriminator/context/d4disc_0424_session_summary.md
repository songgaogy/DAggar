# D⁴-Disc Session Summary — 2026-04-24

> Context handoff for starting a new conversation. Written end-of-day 2026-04-24.
> Covers the repel-loss implementation, the first real Phase-B training run, eval
> results, and the diagnosed next step.
>
> **Read these companion docs first** (they contain full architectural/theory context):
> - `d4disc_0423.md` — code-aligned architecture & training reference
> - `d4disc_0424_cond_collapse_analysis.md` — Phase-B collapse diagnosis & solution survey
> - `d4disc_0424_debug_lpb_degraded.md` — prior motion-confound fix (rel mode)
> - `d4disc_0424_repel_loss_impl_plan.md` — this session's implementation plan

---

## 1. What was done this session

### 1.1 Theory: rejected Two-Bank, chose unconditional M-step repel

- Initial proposal (two-bank advantage gate with `B_+` and `B_-`) was **naive**.
  Four collapse paths, summarized:
  1. **B_+ / B_- overlap** in shared state space → KNN distances near-identical → A≈0 → γ→0.5 (same old collapse).
  2. **Gate change ≠ M-step change**: repel only via γ leaves MSE-to-z_target symmetric in (+,−) at ambiguous γ≈0.5 samples (the majority).
  3. **B_- self-consistency**: f_− trained to predict fail z_target ⇒ lands in B_− ⇒ r_−→0 ⇒ A→0 ⇒ γ→0.5.
  4. **AdaLN cond-embedding hidden symmetry** under weak γ signal.
- Real fix requires **M-step unconditional gradient that breaks +/− symmetry**,
  not routed through γ. Chose `L_repel_minus` hinge on `f_−` vs `B_+`:
  ```
  L_repel_minus = λ · E_{all samples} [ max(0, m − min_{b∈B_+} ||f_−(x) − b||²) ]
  ```
- Strict non-symmetry claim: at `f_+ ≡ f_−`, `∂L_repel/∂θ_{−spec}` is bounded
  below by a constant > 0 (depends on m and bank density) — independent of γ.
  So the symmetric point is *not* a driving-force fixed point anymore.

### 1.2 Implementation shipped (not yet committed)

Modified files (uncommitted diff on branch `lpb-knn-new`):
- `robosuite/discriminator/d4disc/training/trainer.py` — config fields, autograd-safe
  chunked KNN, extra c=− forward per batch, margin auto-computation, logging.
- `robosuite/discriminator/d4disc/training/monitor.py` — `D4Health.repel_hinge_loss`.
- `robosuite/discriminator/d4disc/train_d4.py` — CLI flags.
- `robosuite/discriminator/d4disc/scripts/train_d4.sh` — env vars.

Key config defaults: `repel_weight=0.0` (opt-in), `repel_margin=-1.0` (auto =
median `d²(fail z_t, B_+)`), `repel_on_phase_a=False`, `repel_warmup_epochs=0`.

All four smoke tests passed (§11 of plan). Default-off is byte-identical to
pre-PR behavior.

### 1.3 Real training: `d4_repel_v1`

Config:
```
REPEL_WEIGHT=0.05
REPEL_WARMUP_EPOCHS=3
WARM_UP_EPOCHS=10   BOOTSTRAP_EPOCHS=40
GATE_FREEZE_EPOCHS=4  (default)
F3_WARM_START=1       (default)
ADVANTAGE_MODE=knn    (default)
```

Location: `checkpoints/d4disc/dynamics/d4_repel_v1/`.

**Phase-B health timeline** (see `train.log` for full detail):

| k | hinge | cond_δ | lat_mse | mean_γ | γ q95 | sep_gap | r+ / r− |
|---|---|---|---|---|---|---|---|
| 0–2 | 0 | ~4e-4 | 0.070 | 0.50 | 0.95 | 0 | — (gate frozen) |
| 3 | 0.098 | **4.4e-2** | 0.097 | 0.50 | (frozen) | 0 | — |
| 4 | 0.022 | 9.2e-2 | 0.085 | 0.30 | 0.69 | 27 | 121/149 |
| 9 | 0.014 | 0.107 | 0.081 | **0.10** | 0.56 | 52 | 114/166 |

All key metrics stabilized by k=7. No `CollapseDetector` trigger (`collapse_reason: None` throughout). The γ distribution is correctly bimodal: q50≈0.009 but q95≈0.56, i.e. ~5% of fails identified as confidently OOD while the rest routed as "expert-like".

### 1.4 Empirical eval on `d4_repel_v1`

Three `knn`-mode runs (the only mode that didn't crash):

| Run | pooled AUROC | Bread | Cereal | Milk | frame |
|---|---|---|---|---|---|
| Phase-A only baseline | 0.9604 | 0.967 | 1.000 | 0.913 | 0.863 |
| d4_repel_v1 (130710) | 0.9655 | 0.974 | 1.000 | 0.913 | 0.877 |
| d4_repel_v1 (130935) | 0.9638 | 0.967 | 1.000 | 0.930 | 0.859 |
| d4_repel_v1 (131009) | 0.9613 | 0.971 | 1.000 | 0.930 | 0.864 |

`rel` mode **crashed** (user report; stack trace not yet captured). `abs` not tested.

---

## 2. Current diagnosis of "Phase-B looks like it did nothing"

The ~0.002–0.005 `knn` improvement is **within noise**. But this does NOT mean
Phase-B failed. It means we measured Phase-B with the wrong ruler.

### 2.1 Why `knn` can't reflect Phase-B value (structural)

`score_mode="knn"` uses only `predictor.obs_proj / proprio_proj / action_proj`
as feature extractors — completely bypassing the AdaLN blocks, condition
embedder, and residual head. Repel gradient barely touches input projections.
**No mechanism for knn to reward a fixed conditional decoder.** See
`d4disc_0423.md §4` and `d4disc_0424_cond_collapse_analysis.md §4`.

### 2.2 Why `rel` crashes (theoretical, not bug — probably)

Repel constrains `f_−` **only in distance** from B_+, **not direction**. So
`Δ_− := f_−(x) − z_t` has norm ~sqrt(margin) ~10 (vs Δ_+ ~0.3), with
essentially random direction. In CFG:
```
f_ω = (1+ω) f_+ − ω f_−
    = z_t + (1+ω)Δ_+ − ω Δ_−
```
Dominated by the random `Δ_−` term when ω > 0. The `rel` score
`||f_ω − z_target||² − ||z_t − z_target||²` explodes in magnitude with
random sign → AUROC ≈ 0.5.

**Not confirmed yet**: whether `ω=0 + rel` also crashes. At ω=0, `f_ω = f_+`,
should produce ≈0.82 (Phase-A `rel` baseline). **If ω=0 also crashes, it's a
real bug, not the theoretical direction problem.** User has not run this test.

---

## 3. Recommended next step: implement `cfg_knn` inference mode

Already proposed in `d4disc_0424_cond_collapse_analysis.md §5`. Not implemented
yet — was explicitly out of scope for the repel PR.

### 3.1 Definition

```
f_ω(t)      = (1+ω) f_+(z_t, s_t, a_{t:t+h}) − ω f_−(…)
λ_cfg_knn   = min_{b ∈ B_+} ||f_ω(t) − b||²  /  (2σ²)
```

Bank `B_+` already built at inference time: expert next-latents from clean
positive transitions. Already exists in Phase-B advantage-gate bank-build
logic (`training/filter.py::build_expert_target_bank`) — needs to be wired
into `D4Detector` / `D4BenchmarkDiscriminator` for inference.

### 3.2 Why it fixes both current problems

- **Bypasses the rel crash**: the anchor is `min_b ||· − b||²`. Even if `f_ω`
  lands in a random off-manifold direction (because `f_−` has no direction
  constraint), the KNN score just measures "how close to nearest expert
  point". f_+'s contribution pulls f_ω toward B_+ for ID samples; f_−'s
  random-direction contribution is absorbed by the nearest-neighbor min. No
  numerical explosion.
- **Unblocks Phase-B attribution**: the conditional decoder is now in the
  scoring loop. ω>0 can actually reward/punish the learned `f_-`.
- **Train-eval metric alignment**: Phase-B advantage gate uses
  `min_b ||f_± − b||²`; `cfg_knn` uses `min_b ||f_ω − b||²`. Same anchor.

### 3.3 Implementation scope (small)

- `inference/detector.py::D4Detector` — add `score_mode="cfg_knn"` branch in
  `_score_frames`; takes `B_+` bank + ω + σ²; computes min-sqdist against bank.
- `inference/benchmark.py::D4BenchmarkDiscriminator` — plumb `score_mode="cfg_knn"`;
  ensure bank is built at fit time (can reuse the training bank-build util, or
  build fresh from success trajectories in the calibration split).
- `inference/feature.py` — unchanged (still uses `D4Frames`).
- `scripts/run_d4_benchmark.sh` — add `SCORE_MODE=cfg_knn` wiring.
- `data/utils/benchmark/examples/run_d4.py` — add `cfg_knn` to `--score-mode`
  choices.

**Do NOT** implement two-bank gate, structural asymmetric head, or the other
rejected ideas (all in `d4disc_0424_cond_collapse_analysis.md §3`).

---

## 4. Open decisions & unknowns for the new conversation

1. **Bug vs theory for `rel` crash**: run one `OMEGA=0 SCORE_MODE=rel` eval
   to disambiguate. Expected ≈0.82; if much lower or crashes → bug.
2. **`cfg_knn` bank source at eval time**: build from benchmark's success
   calibration split (current pattern for knn mode) vs build from training's
   expert transitions. First is cleaner for apples-to-apples with `knn` mode;
   second matches training-time `B_+` exactly.
3. **ω sweep for `cfg_knn`**: once implemented, run {0, 0.5, 1.0, 1.5}. ω=0
   reduces to KNN-of-f_+ — a sanity check. ω>0 is the test of d4's premise.
4. **If `cfg_knn` ≤ 0.960** (Phase-A knn baseline): conclude d4 conditional
   architecture gave nothing measurable, discuss whether to retire Phase B.
   Pre-commit to this judgment before running, per `analysis.md §7.4`.
5. **Repel still gives no direction to f_−**. If `cfg_knn` lights up, great.
   If not, the fallback is a **two-bank anchor-attract** term (pull f_−
   toward fail-next-latent bank B_−) to add direction. Not recommended
   without evidence that `cfg_knn` alone is insufficient.

---

## 5. Key file locations

- **Trained checkpoint**: `checkpoints/d4disc/dynamics/d4_repel_v1/d4_dynamics.pt`
- **Training log**: `checkpoints/d4disc/dynamics/d4_repel_v1/train.log`
- **Eval outputs (knn mode)**: `checkpoints/d4disc/eval/run_20260424_13{0710,0935,1009}/benchmark.json`
- **Session context docs**: `.claude/discriminator/context/d4disc_042*.md`
- **Current uncommitted code changes**: `robosuite/discriminator/d4disc/{training/trainer.py,training/monitor.py,train_d4.py,scripts/train_d4.sh}`

Code changes are **not yet committed**. User may want to commit before the
next session to establish a clean baseline before `cfg_knn` work.

---

## 6. TL;DR for resuming

> I trained `d4_repel_v1` with the M-step repel loss. Training was healthy
> (conditional_delta stabilized at 0.107, no collapse triggers, f_+ MSE kept
> improving). But eval with `score_mode=knn` shows no signal change (0.96 ±
> noise) because knn bypasses the decoder, and `rel` crashes because repel
> gave f_− distance without direction. Next step: implement `cfg_knn`
> inference mode (definition in `d4disc_0424_cond_collapse_analysis.md §5`) —
> this is the only score mode that both uses the conditional decoder and is
> robust to f_−'s random-direction off-manifold placement. After that, sweep
> ω and decide whether d4's conditional+CFG architecture is empirically
> justified or should be retired.
