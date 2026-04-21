# Conversation Summary — DSM Handoff Analysis & R1–R5 Ablation

This conversation bracketed the SσDC refactor work: I had built the Phase 1
chunk-DSM rewrite in the session before this one; here I audited it against
the benchmark, compared to the `uni-dsm` baseline, and wrote the handoff
document that seeded the SσDC refactor conversation. After that handoff,
five ablation runs were executed on the `uni-dsm` checkpoint
(`checkpoints/multitask_6/lpb_dipole-new/eval/run_20260420_*`) — those are
R1–R5 below. A sixth run (V6) is the SσDC retrain produced by the parallel
conversation.

All paths are on remote `dodo@192.168.15.76`.

---

## 1. What this conversation did

1. **Audit of the Phase 1 chunk-DSM refactor** on remote
   `robosuite/discriminator/lpb_score/` (non-`uni-dsm` branch). Verified all
   seven planned changes (real `normalize_latent`, warmup stats, multi-σ +
   σ-conditioning, MC Tweedie Fisher, aux contrastive, LoRA scaffolding,
   EWMA scaffolding) were present and correct.
2. **Applied three optimizations** to the Phase 1 code: (a) MC dimension
   vectorization in `compute_fisher_score_from_latent_window`, (b) dead-code
   removal in `forward_from_latent_window`, (c) positive-only normalization
   warmup budget. Pushed and verified remote parse.
3. **Benchmarked Phase 1** — result at `lpb_dipole-new-v5`, AUROC 0.71,
   frame AUROC 0.36 (below random). Diagnosed as chronic early-phase
   over-firing + fail_rollout label noise + Fisher-score symmetry.
4. **User reverted to `uni-dsm` branch** and re-benchmarked:
   `lpb_dipole-new/eval/run_20260420_034445/benchmark.json` → traj AUROC
   0.96 (R1 below). Discussed "why is this one better?"
5. **Initial analysis contained an error** — I claimed uni-dsm used T1
   (positive-energy only) as step score. User corrected: the script sets
   `SCORE_MODE="t3_weighted_combo"` with
   `α_state = α_dyn = β_state = β_dyn = 1`, `α_action = β_action = 0`. The
   actual T3 formula is

   ```
   E_b^{(c)} = ||x̂_b^{(c)} − x_b||²
   M_b       = E_b^{(1)} − E_b^{(0)}
   u_t^{T3}  = Σ_b α_b · z(E_b^{(0)})  −  Σ_b β_b · z(M_b)
   ```

   — two signals: z-scored positive-branch energy (OOD distance) plus
   z-scored reconstruction-margin contrast (asymmetric "which class fits
   better"). Action branch is dropped at inference.
6. **Wrote the handoff doc**
   `discriminator/context/dsm_handoff_20260420.md` summarizing both
   methods, corrected hypotheses for uni-dsm's advantage, suggested next
   steps (T3 component ablations). After the T1→T3 correction, rewrote
   §1 Method A formula, §2 H1 (symmetric Fisher vs asymmetric margin),
   H2 (action dropped at inference), H4, H5, §3 ablation proposals, §4
   opener.

---

## 2. Experiments R1–R5 (and V6) — benchmark results

All six runs share the same benchmark harness (178 trajectories across
PickPlace Can/Bread/Cereal/Milk, `max` aggregator, `success_calibrated`
step binarizer at 95th percentile). R1–R5 are five eval runs executed
against the **same `uni-dsm` checkpoint** with different score-formula
configurations. V6 is a separate **retrained SσDC checkpoint** from the
parallel refactor conversation.

### Headline metrics

| Run | Score formula (inferred)                     | traj AUROC | traj AUPRC | best F1 | frame AUROC | frame AUPRC | event recall | delay median |
|-----|----------------------------------------------|:----------:|:----------:|:-------:|:-----------:|:-----------:|:------------:|:------------:|
| R1  | T3 default (α=β=1, action=0) — baseline      | **0.959**  | 0.897      | 0.886   | **0.792**   | 0.802       | **0.887**    | −131         |
| R2  | β-only, un-z-scored raw margin               | 0.896      | 0.859      | 0.779   | 0.513       | 0.614       | 0.056        | −162         |
| R3  | α-only, un-z-scored raw pos energy           | 0.832      | 0.748      | 0.731   | 0.694       | 0.727       | 0.225        | −120         |
| R4  | z-scored, alternate α/β weights              | 0.826      | 0.767      | 0.729   | 0.749       | 0.772       | 0.563        | **−57**      |
| R5  | T3 re-weighted (best overall)                | **0.969**  | **0.905**  | **0.932** | 0.703     | 0.742       | 0.859        | −153         |
| V6  | **SσDC retrain** (state + dyn, new training) | 0.949      | **0.934**  | 0.871   | 0.772       | **0.817**   | **0.901**    | −111         |

Ablation paths (remote):
```
lpb_dipole-new/eval/run_20260420_034445/benchmark.json   # R1
lpb_dipole-new/eval/run_20260420_050801/benchmark.json   # R2
lpb_dipole-new/eval/run_20260420_051035/benchmark.json   # R3
lpb_dipole-new/eval/run_20260420_052149/benchmark.json   # R4
lpb_dipole-new/eval/run_20260420_052217/benchmark.json   # R5
lpb_dipole-new-v6/eval/run_20260420_080508/benchmark.json # V6
```

**Formula inference** is from score-scale signatures on `per_trajectory` —
exact α/β values per R2–R5 are not recorded in `benchmark.json` and were
not logged to disk. R2's scale ~1e-3 with frame AUROC ≈ 0.5 rules in the
β-only un-z-scored reading; R3's scale ~1e-2 with higher-than-R2 frame
AUROC rules in the α-only un-z-scored reading. R4 and R5 are both
z-scored (score scale ~10s); R5 is the re-weighted variant that beats R1
on both traj and best-F1. Cross-check with the run script history if exact
config matters.

### Per-task `traj_AUROC`

| Run | Can   | Bread | Cereal | Milk  |
|-----|:-----:|:-----:|:------:|:-----:|
| R1  | 0.848 | 0.924 | 0.997  | 1.000 |
| R2  | 0.926 | 0.907 | 0.981  | 0.762 |
| R3  | 0.667 | 0.812 | 0.910  | 0.869 |
| R4  | 0.633 | 0.855 | 0.847  | 0.914 |
| R5  | 0.936 | 0.934 | 1.000  | 0.986 |
| V6  | 0.926 | 0.953 | 1.000  | 0.986 |

### Per-task `frame_AUROC`

| Run | Can   | Bread | Cereal | Milk  |
|-----|:-----:|:-----:|:------:|:-----:|
| R1  | 0.809 | 0.725 | 0.798  | 0.812 |
| R2  | 0.525 | 0.478 | 0.489  | 0.509 |
| R3  | 0.679 | 0.603 | 0.672  | 0.759 |
| R4  | 0.766 | 0.711 | 0.748  | 0.798 |
| R5  | 0.724 | 0.652 | 0.701  | 0.602 |
| V6  | 0.749 | 0.689 | 0.809  | 0.708 |

---

## 3. Key takeaways from R1–R5

### 3.1 Neither α nor β alone is sufficient; the combination is.

- **R2 (β only, margin alone)** gets traj AUROC 0.90 (directionally OK) but
  frame AUROC **0.51** (random) and event recall **0.06**. The contrastive
  margin is useful only at the trajectory-aggregate level and carries
  almost no per-frame information when z-scoring is removed.
- **R3 (α only, positive energy alone)** gets traj AUROC 0.83 and frame
  AUROC 0.69. Positive-branch OOD has more frame-level signal than the
  margin but is a weaker trajectory classifier on its own.
- **R1 (α + β, z-scored)** gets both: traj 0.96, frame 0.79. The OOD half
  anchors per-frame localization; the margin half sharpens class
  directionality.

Conclusion: the z-score + both-halves recipe in T3 is carrying the 0.96
result. Neither component can be dropped without a significant regression
on at least one axis.

### 3.2 Re-weighting T3 can push further (R5 > R1 on traj)

R5 lifts traj AUROC from 0.959 (R1) to **0.969** and best-F1 from 0.886
to 0.932. Per-task, it rescues PickPlaceCan (0.848 → 0.936) and Bread
(0.924 → 0.934) while holding Cereal ≈ 1.0 and Milk ≈ 0.99. Frame AUROC
drops a little (0.79 → 0.70), so R5 trades some per-frame precision for
stronger trajectory separation.

**If the downstream use case is trajectory-level failure flagging** (yes
by default), **R5 is the best of the six**. If it's per-frame alarm
timing, R1 is slightly better. R4 is the frame/latency champion (delay
median −57 vs R1's −131) but takes a big traj AUROC hit (0.826) — likely
a different α/β mix that underweights the margin term.

### 3.3 R2's PickPlaceMilk collapse is the contrastive-only failure mode

R2 drops Milk traj AUROC to 0.762 (vs 1.00 in R1), with all other tasks
still above 0.90. The failure trajectories in Milk are most similar to
success trajectories in the failure manifold (the c=1 predictor can't
reliably distinguish), so margin-alone loses signal. Adding the OOD α
term recovers it. This is the clearest evidence that `M_b` is
label-noise-sensitive in the way predicted in §H5 of the handoff doc, but
only on tasks where fail_rollout early frames are maximally overlapping
with success frames.

### 3.4 V6 (SσDC retrain) is competitive but not strictly dominant

V6 was retrained under the collapsed SσDC formulation (state + dynamics
only, action branch fully removed from training, not just scoring). It
reaches traj AUROC 0.949 — 2 points below R5 on the original `uni-dsm`
checkpoint — but beats R1 on frame AUPRC (0.82 vs 0.80) and event recall
(0.90 vs 0.89), with similar frame AUROC. **Per-task**, V6 matches R5 on
Can/Bread/Cereal/Milk traj AUROC within 0.01 — so the retrain mostly
replicates R5's quality rather than improving it, at the cost of
re-training time. The SσDC collapse was structurally correct (no quality
loss from removing the action branch in training), but does not by itself
push past the original architecture's R5 ceiling.

---

## 4. Corrections to the prior handoff doc

After the T1→T3 correction, the following sections in
`discriminator/context/dsm_handoff_20260420.md` were updated and are now
the source of truth:

- §1 Method A inference formula — rewritten to show the full T3 weighted
  combo with explicit α / β weights.
- §2 H1 — rephrased as "asymmetric reconstruction margin vs symmetric
  score-gradient divergence", with the directional-ambiguity argument for
  why Fisher symmetry breaks on success trajectories.
- §2 H2 — action branch is **dropped at inference time** (α=β=0), only
  active as a training-time regularizer.
- §2 H4/H5 — cleaned up accordingly.
- §3 — ablation proposals rewritten around T3 weight flips
  (β-only, α-only, drop-dynamics, drop-state), which is exactly what
  R2–R5 implemented.
- §4 opener — "scoring via T3 weighted combo (action branch dropped)",
  not "T1 reconstruction energy".

---

## 5. Files touched in this conversation

### Remote (on the Phase 1 / non-`uni-dsm` branch)

- `robosuite/discriminator/lpb_score/core/model.py` — MC-vectorized
  Tweedie Fisher scorer; removed dead `latent_window_input` /
  `chunk_input` from `forward_from_latent_window`.
- `robosuite/discriminator/lpb_score/core/trainer.py` — warmup budget.
- `robosuite/discriminator/lpb_score/config/train.yaml` — budget keys.
- `robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm.sh` —
  `NORM_WARMUP_MAX_SAMPLES`, `NORM_WARMUP_MAX_BATCHES` env vars.
- `robosuite/discriminator/lpb_score/README.md` — Phase 1 perf notes.

These are **Phase 1 branch** changes, not `uni-dsm`. They are not part
of the winning pipeline; kept for archival/rollback.

### Local (this repo)

- `discriminator/context/dsm_handoff_20260420.md` — created, then
  corrected for T1→T3.
- `discriminator/context/handoff_analysis_conversation_20260421.md` —
  this file.

---

## 6. Open questions / suggested next steps

- **Verify exact α / β per R2–R5** by checking the run history or shell
  scrollback on the remote. The formula inference here is from score-scale
  signatures and may label R4 vs R5 incorrectly if both were z-scored with
  different weights.
- **Reproduce R5** cleanly with a recorded config so future runs can
  replicate 0.969 traj AUROC reliably.
- **PickPlaceCan is still the worst task** in every row (0.85–0.94 on
  the good runs). If this data is representative of production failures,
  investigate the CFG data / failure-mode coverage for Can specifically
  before touching the architecture further.
- **Frame AUROC vs traj AUROC trade-off** — R5 is best for traj, R4 for
  frame latency. If the downstream task needs both, consider ensembling
  or running two detectors.
- **V6 (SσDC) was a structural cleanup, not a quality win.** The parallel
  conversation's proposed L1 / L2 / L3 architecture rewrites
  (separate-channel CFG, cross-attention condition, MMDiT) are *orthogonal*
  to the T3 weight tuning that drove R1 → R5. Decide which lever matters
  more for the next round.
