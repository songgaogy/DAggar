# Conversation Summary — V6 Post-hoc Sweep + σ × Loss Ablation Training

This conversation started from the two prior handoff docs
(`analysis_on_previous_conversation.md` + `sgdc_refactor_conversation_summary.md`).
It did three things:

1. **Post-hoc eval sweep on the V6 checkpoint** — 18 benchmark runs across
   `(lambda_mode, window, α, β, delta)`. Found the Pareto configs and
   discovered the `delta` knob is a no-op.
2. **Consolidated results → `robosuite/discriminator/lpb_score/results/`**
   (README + σ × loss ablation plan).
3. **Launched 7-run σ × loss training ablation** with EMA on,
   dropout=0.1, per-trajectory fail labels preserved. Wrote the
   training launcher and the evaluation launcher. Training finished;
   eval is **pending** at the time of this handoff.

Branch: `uni-dsm` (no checkpoint-breaking code changes this session —
all conceptual work is in `results/` and `scripts/`).

---

## 1. V6 checkpoint — fixed reference point

**Path (local):**
`checkpoints/multitask_6/lpb_dipole-new-v6/lpb_dipole_dsm_20260420_070130/lpb_dipole_dsm_20260420_070130.pt`

**Actual training parameters** (per `scripts/train_lpb_score_dsm.sh`,
which overrides `config/train.yaml`):

- `num_pos_traj=200`, `num_neg_traj=100` (per task)
- `transition_horizon=2`, `batch_size=1024`, `positive_ratio=0.7`
- `noise_scale=0.08`, `state_loss_weight=1.0`, `dynamics_loss_weight=1.0`
- `dropout=0.0`, `use_ema=false`  ← both are fixed ON in the ablation
- 50 epochs, LR 2e-4, AdamW, grad_clip_norm 1.0

(`results/README.md` has the full param dump for reproduction.)

---

## 2. Post-hoc sweep findings (3 rounds on V6)

### 2.1 Sweeps executed

- `eval/run_20260421_134617/134625/163110/` — 3 single-variable runs
  (`lambda_window_size ∈ {10, 20, -1}`)
- `eval/sweep_20260421_170146/` — 10-case sweep (windows 10/20/30/40 ×
  mean/max, α∈{0,0.5,1}×β∈{0,0.5,1} grid)
- `eval/sweep_tail_20260421_174845/` — 8-case tail (max windows
  60/80/120, mean windows 50/60/80, and 2 stricter-DELTA cases that
  turned out to be dead)

### 2.2 Pareto-optimal configs (all β-only, α dropped)

| Config | mode | window | β_s | β_d | traj_AUROC | frame_AUROC | evP | evR | delay_med | Best for |
|---|---|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|---|
| **A**  | mean | 50  | 1 | 1 | **0.9728** | 0.8422 | 0.610 | 0.930 | -79 | traj ranking |
| **B**  | max  | -1  | 1 | 1 | 0.9493 | **0.9228** | **0.984** | 0.845 | **-7** | frame/event (default) |
| **C**  | max  | 120 | 1 | 1 | 0.9493 | 0.8938 | 0.942 | 0.887 | -13 | bounded-memory online |

**Recommended default: Config B** (max + cumulative-max gives
near-perfect event_precision and detection delay ≈ GT at the cost of 0.024
traj AUROC). Full per-task breakdown in `results/README.md`.

### 2.3 Structural discoveries

- **Trajectory ranking and frame localization have different optimal
  aggregators.** `mean/50` maximizes traj AUROC via magnitude
  comparison; `max/-1` (cumulative max) gives one-shot threshold
  crossing, which is frame/event-optimal. Dual-λ was proposed
  (行动 1) but user declined.
- **α branch carries no signal at σ=0.08.** Every α>0 case in the
  sweep underperformed β-only. This is the V6-specific data point that
  motivated the σ-sweep ablation (see §4).
- **The `DELTA` CLI is a dead knob in this benchmark harness.** The
  benchmark sets its own `step_success_percentile=95.0` independent of
  the detector-internal threshold. This lives in
  `data/utils/benchmark/examples/run_lpb_score.py` (**gitignored, do
  not auto-edit**). To truly change step binarization, edit that file
  locally.
- `lambda_mode=max` collapses all windows to the same traj_AUROC
  (0.9493) because the harness uses `top-k mean` for trajectory score,
  which devolves to `max(step_scores)` when λ is already monotone.
  Windows still differ on frame/event metrics.

---

## 3. Math answer locked in — α branch is optional

Under the SσDC identity `-log p_σ(x|c) = (1/2σ²)·E[‖g(x+σε,c)−x‖²] +
C(x,σ)`, the log-likelihood ratio is
`LLR ∝ (E⁽¹⁾ − E⁽⁰⁾)`, which is exactly the **β term** (up to a
per-task affine transform from z-scoring). β-only is the Bayes-optimal
classifier; α is an OOD-prior hedge that only activates when the c=1
branch collapses or when the σ is too small to give discriminative E⁺.

**Decision:** α is dropped from all production configs. It stays in
the model (scoring code still supports it for diagnostics) but
defaults to 0. See `results/README.md` §1 for the recommended configs.

---

## 4. σ × loss ablation — plan + what's trained

Plan in `results/ablation_plan_sigma_loss.md`. Two-phase design, 7
runs total.

**All runs lock in:** EMA on (decay 0.999), dropout=0.1,
per-trajectory `traj_type` (user forbade frame-level labels).

### 4.1 Run table (all trained, checkpoints on disk)

Location: `checkpoints/multitask_6/lpb_dipole-new-v6/abl_sig_loss_20260421_203614/`
(the abl script's default `BASE_SAVE_DIR` is
`lpb_score_new-v6` — user renamed it before launch; see `scripts/train_lpb_score_dsm_abl.sh:64`.)

| case | σ | state_lw | dyn_lw | checkpoint |
|---|:-:|:-:|:-:|---|
| A1 | 0.04 | 1.0 | 1.0 | `A1__sig-0.04__lw-1.0-1.0/lpb_score_dsm_A1.pt` |
| A2 | 0.08 | 1.0 | 1.0 | `A2__sig-0.08__lw-1.0-1.0/lpb_score_dsm_A2.pt` |
| A3 | 0.12 | 1.0 | 1.0 | `A3__sig-0.12__lw-1.0-1.0/lpb_score_dsm_A3.pt` |
| A4 | 0.20 | 1.0 | 1.0 | `A4__sig-0.20__lw-1.0-1.0/lpb_score_dsm_A4.pt` |
| B1 | 0.08 | 1.0 | 0.0 | `B1__sig-0.08__lw-1.0-0.0/lpb_score_dsm_B1.pt` (state-only) |
| B2 | 0.08 | 0.0 | 1.0 | `B2__sig-0.08__lw-0.0-1.0/lpb_score_dsm_B2.pt` (dyn-only) |
| B3 | 0.08 | 1.0 | 1.0 | `B3__sig-0.08__lw-1.0-1.0/lpb_score_dsm_B3.pt` (dup of A2) |

Each run dir also has `console.log` + `lpb_score_dsm_<case>_ep{0010..0050}.pt`
periodic checkpoints.

### 4.2 Evaluation — NOT YET RUN

Script: `robosuite/discriminator/lpb_score/scripts/eval_lpb_score_dsm_abl.sh`.
Plan: 18 benchmark jobs.

```
A1-A4 × {cfgB, cfgA, cfgB_alpha}  = 12 jobs
B1     × {cfgB_state, cfgA_state} =  2 jobs  (β_s=1, β_d=0)
B2     × {cfgB_dyn,   cfgA_dyn}   =  2 jobs  (β_s=0, β_d=1)
B3     × {cfgB, cfgA}             =  2 jobs
                                     18 total
```

Where:
- `cfgB` = max, win=-1, β=(1,1), α=(0,0)  — Config B recipe
- `cfgA` = mean, win=50, β=(1,1), α=(0,0) — Config A recipe
- `cfgB_alpha` = max, win=-1, β=(1,1), α=(1,1) — **σ-vs-α diagnostic**:
  verifies whether a larger σ revived the α branch (the Phase A
  question)
- `cfgB_state/A_state` / `cfgB_dyn/A_dyn` = match each B-run's trained head

**Launch command (user runs):**
```bash
SWEEP_DIR=checkpoints/multitask_6/lpb_dipole-new-v6/abl_sig_loss_20260421_203614 \
  bash robosuite/discriminator/lpb_score/scripts/eval_lpb_score_dsm_abl.sh
```

Output:
`${SWEEP_DIR}/eval_<ts>/<case>__<cfg>/benchmark.json` plus
`eval_summary.csv` (auto-aggregated, 18 rows, printed at end).

### 4.3 Success criteria to apply when results land

From `ablation_plan_sigma_loss.md §3`:

- **σ\* pick:** argmax `frame_AUROC` under cfgB across A1-A4;
  accept σ\* ≠ 0.08 only if improvement ≥ 0.01.
- **EMA/dropout effect:** A2 vs V6 isolates this (same σ, same
  loss, only EMA+dropout differ). V6 reference under cfgB is
  traj=0.9493, frame=0.9228. Expected +0.002–0.005 traj.
- **α revival:** scan `cfgB_alpha` across A1-A4. Success = any σ
  where `α=1` config beats `α=0` cfgB by ≥ 0.01 frame AUROC. If
  none, α is confirmed dead and can be dropped from training too.
- **Branch necessity (Phase B):** B3 (both) vs B1/B2. If either
  alone lands within 0.01 traj / 0.02 frame of B3, the other branch
  is under-contributing — structural simplification candidate. If
  both drop ≥ 0.05, factor complementarity is confirmed.

---

## 5. Files touched this conversation

### Created

- `robosuite/discriminator/lpb_score/results/README.md` —
  V6 perf + recommended Config A/B/C + training params for reproduction.
- `robosuite/discriminator/lpb_score/results/ablation_plan_sigma_loss.md` —
  Phase A × B design, success criteria, results template.
- `robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm_abl.sh` —
  7-run parallel launcher; tunables: `MAX_TASK_PRL`, `GPU_IDS`,
  `SIGMA_STAR`, `CASES`, `DRY_RUN`, etc.
- `robosuite/discriminator/lpb_score/scripts/eval_lpb_score_dsm_abl.sh` —
  18-job eval launcher; auto-aggregates to CSV.
- `.claude/discriminator/context/post_sweep_ablation_conversation_20260421.md` — this file.

### Modified

- `scripts/run_lpb_score_benchmark.sh` —
  `DELTA="5"` → `DELTA="${DELTA:-5}"` so the sweep script can override
  (even though we later discovered DELTA is a dead knob in the harness).
- `scripts/sweep_lpb_score_benchmark.sh` — replaced the original 10-case
  grid with 8 tail cases (max windows 60/80/120, mean 50/60/80, 2 DELTA
  variants). Added a 9th `delta` positional arg to `run_case`.

### Not modified (but flagged)

- `data/utils/benchmark/examples/run_lpb_score.py` — gitignored; owns
  `step_success_percentile` which is the real step-binarization knob.
  Left alone per gitignore discipline.
- `core/dataset.py` — user forbade frame-level label denoising. Per-trajectory
  `traj_type` semantics preserved.
- `core/dsm_discriminator.py` — Dual-λ (separate λ_frame / λ_traj) was
  proposed; user declined. Single-λ via env still.

---

## 6. Open decisions when eval lands

1. **Pick σ\*** from Phase A. Update `results/ablation_plan_sigma_loss.md §6`
   results template. If σ\* ≠ 0.08, rerun Phase B with the new σ
   (`CASES="B1 B2 B3" SIGMA_STAR=<σ*> bash train_lpb_score_dsm_abl.sh`).
2. **Decide on α.** If cfgB_alpha never beats cfgB, close the chapter on
   α. Otherwise add α to Config B as a weighted add-on at inference.
3. **Promote the winner.** The best ablation checkpoint should become
   the new default in `scripts/run_lpb_score_benchmark.sh` (update
   `DSM_CKPT`). Also update `results/README.md` with the V7 numbers.
4. **Ablation next round.** If σ\* ≠ 0.08 and there's signal of further
   improvement, run a denser σ grid around σ\*. If Phase B shows one
   branch dominates, consider architectural simplification (drop the
   weaker head, or reallocate its capacity).

---

## 7. Hard constraints locked in by user (do not violate)

- **No frame-level failure labels.** Per-trajectory `traj_type` only.
- **Do not edit `data/utils/benchmark/examples/run_lpb_score.py`** — it's
  gitignored.
- **EMA + dropout=0.1** are permanent from here on.
- **α is dropped in deployed configs.** Only kept in model code for
  diagnostics; scoring defaults `α=0`.
- **Dual-λ (separate aggregators for traj vs frame) was declined.** Stick
  to single-λ via env override.
