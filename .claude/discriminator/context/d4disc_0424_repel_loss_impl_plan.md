# D⁴-Disc: Phase-B M-step Repel Loss (Implementation Plan)

> **Date**: 2026-04-24
> **Scope**: Add an unconditional M-step symmetry-breaking term `L_repel_minus`
> to prevent the Phase-B conditional collapse (`f_+ ≡ f_-`). This is the only
> change in this PR.
> **Companion docs (read first for full context)**:
> - `d4disc_0423.md` — code-aligned architecture & training reference
> - `d4disc_0424_cond_collapse_analysis.md` — problem diagnosis & solution survey
> - `d4disc_0424_debug_lpb_degraded.md` — prior motion-confound fix
>
> **Do NOT implement**: Two-bank advantage gate, `cfg_knn` inference mode,
> structural asymmetric head, InfoGAN / Kingma M2 decoupled classifier. Those
> are out of scope for this PR; keep the existing advantage gate logic
> (`advantage_mode="knn"` with single bank `B_+`) untouched.

---

## 1. What to implement (one sentence)

Add a loss term applied during Phase-B training that, on **every batch
element**, pushes the `c=-` branch prediction AWAY from the expert-latent bank
`B_+` via a hinge:

```
L_repel_minus  =  λ_rep · E_{j ∈ batch} [ max(0, m - min_{b∈B_+} ||f_θ(x_j | c=-) - b||²) ]
```

where `B_+` is the existing `self._expert_z_bank` already built by
`training/trainer.py::D4Trainer.run()` for the KNN advantage gate.

This is added **on top of** the existing latent + proprio MSE loss. It is **not
a replacement** for any existing term.

---

## 2. Why this term and not another

See `d4disc_0424_cond_collapse_analysis.md §3.3` and the follow-up discussion
(conversation context). TL;DR:

- Any symmetry-breaker that only modifies γ (the E-step) fails: at `f_+ ≡ f_-`
  the M-step MSE gradient on `(θ_{+spec} - θ_{-spec})` vanishes for all samples
  with γ≈0.5 (which is most of them due to `B_+` / `B_-` overlap in state space).
- `L_repel_minus` puts an **unconditional** gradient on `θ_{-spec}` at every
  batch — it does not route through γ, so it is robust to any degeneracy in
  the gate's signal.
- Semantically aligns with CFG at inference: `f_-` near the "off-expert"
  region ⇒ `f_ω = (1+ω)f_+ - ω f_-` amplifies away from the bank for OOD.

---

## 3. Files to modify

1. `robosuite/discriminator/d4disc/training/trainer.py` — config fields,
   margin auto-computation, extra forward + hinge in `_run_step`, logging
   plumbing.
2. `robosuite/discriminator/d4disc/training/monitor.py` — add
   `repel_hinge_loss` field to `D4Health`.
3. `robosuite/discriminator/d4disc/train_d4.py` — CLI flags.
4. `robosuite/discriminator/d4disc/scripts/train_d4.sh` — env vars + arg
   threading.

Do **not** touch:
- `training/filter.py` (bank building and gate logic stay as-is; `B_+` is
  already built once per Phase-B at `trainer.py:507-521`, reuse it directly).
- `inference/*` (no inference changes this PR).
- `models/*` (no architectural change).

---

## 4. Config fields (in `D4TrainerConfig`, `trainer.py`)

Add these fields to `D4TrainerConfig` with sensible defaults. All are opt-in;
default `repel_weight=0.0` makes this a pure no-op until the user enables it.

```python
# Phase-B M-step repel loss (anchor c=- away from B_+ expert bank).
# See d4disc_0424_repel_loss_impl_plan.md.
repel_weight: float = 0.0        # 0.0 = disabled; recommended first try 0.1
repel_margin: float = -1.0       # <0 means auto from bank diagnostics
repel_margin_percentile: float = 0.5  # which quantile of d²(fail z_t, B_+) to use when auto
repel_on_phase_a: bool = False   # default: Phase-B only; True = also active in Phase A
repel_warmup_epochs: int = 0     # number of Phase-B epochs to skip repel (0 = start immediately)
```

Notes:
- `repel_margin < 0` ⇒ auto: margin computed once after `B_+` is built
  (see §6 below). If the user passes a positive `repel_margin`, use it verbatim.
- `repel_on_phase_a`: **default False**. Phase A currently trains only `c=+`;
  turning this on would cold-start the `-` branch during Phase A warm-up
  against `B_+`. Optional exploration; not default.
- `repel_warmup_epochs`: default 0. Harmless to leave at 0 (analysis doc §3.3
  suggests this is optional; the hinge is self-gating via `max(0, ...)`).

---

## 5. `D4Health` field (in `training/monitor.py`)

Add one field to the dataclass:

```python
@dataclass
class D4Health:
    ...
    repel_hinge_loss: float = 0.0   # mean hinge value across the last epoch (0 when disabled)
```

Do **not** change `CollapseDetector` logic. Repel loss prevents collapse, it
does not need its own trigger.

---

## 6. Margin auto-computation (in `trainer.py::run()`)

When `cfg.repel_weight > 0` and `cfg.repel_margin < 0`, compute the margin
**once**, right after `build_expert_target_bank(...)` at approximately
`trainer.py:508-521`.

Compute `d²(z_t of fail samples, B_+)` using `knn_sqdist` from
`d3disc.filter` — this is exactly the quantity `warm_start_gamma_from_knn`
already produces as `d2_q50`. Two acceptable implementations:

**Option A (preferred, simpler)**: if `cfg.f3_warm_start=True` and the
warm-start was just executed, read `d2_q50` from the warm-start diagnostic
dict that is already appended to `self.history["warm_start"]`. Pull the
most recent entry and use its `d2_q50` or `d2_q<percentile>` field.

**Option B (fallback, robust)**: write a small helper
`_compute_repel_margin(bank, dataset, encoder, percentile) -> float`:
- encode a subsample of fail-sample `current_image` through the frozen encoder
- call `knn_sqdist(fail_z_t, bank, k=1)`
- return `torch.quantile(d2, percentile).item()`

Store the resolved margin on `self._repel_margin` (a float) for use in
`_run_step`. Print it to the log:

```
[d4_trainer] repel loss active: weight=0.10 margin=<value> (auto, p=0.5)
```

If `cfg.repel_weight <= 0`, skip the computation entirely and set
`self._repel_margin = None`.

---

## 7. The actual loss term (in `trainer.py::_run_step`)

Current `_run_step` does one predictor forward with `cond_idx=cond` (sampled
per-element by phase). For the repel loss, add a **second forward** on the
same batch with `cond_idx=ALL_MINUS`, feed its `pred_latent` through
`knn_sqdist` against `self._expert_z_bank`, and add the hinge to `loss`.

Sketch (integrate cleanly into the existing `_dtype_context()` / grad-enabled
block; do not duplicate encoder work — reuse `z_t`, proprio, action):

```python
# Inside _run_step, after computing latent_mse and proprio_mse and the base loss:
repel_hinge_val = 0.0
repel_active = (
    train
    and self.cfg.repel_weight > 0.0
    and getattr(self, "_expert_z_bank", None) is not None
    and getattr(self, "_repel_margin", None) is not None
    and (phase == "B" or self.cfg.repel_on_phase_a)
    and (phase != "B" or self._current_bootstrap_k >= self.cfg.repel_warmup_epochs)
)
if repel_active:
    batch_size_now = int(z_t.shape[0])
    c_minus_all = torch.full(
        (batch_size_now,), ConditionEmbedder.COND_MINUS,
        dtype=torch.long, device=self.device,
    )
    out_minus_all = self.predictor(
        z_t, b["current_proprio"], b["action_sequence"], cond_idx=c_minus_all,
    )
    r_minus_to_pos = knn_sqdist(
        out_minus_all["pred_latent"],
        self._expert_z_bank,
        k=int(self.cfg.advantage_knn_k),
        chunk_size=int(self.cfg.advantage_knn_chunk_size),
    )
    repel_hinge = torch.clamp(self._repel_margin - r_minus_to_pos, min=0.0).mean()
    loss = loss + float(self.cfg.repel_weight) * repel_hinge
    repel_hinge_val = float(repel_hinge.detach().item())
```

Then backward / step as usual.

Important details:
- `knn_sqdist` import: `from robosuite.discriminator.d3disc.filter import knn_sqdist`
  (already imported at `filter.py:11`; re-import in `trainer.py` if needed).
- **Gradient DOES flow** through the second predictor forward — that is the
  whole point. Do **not** wrap it in `torch.no_grad()`. But the `knn_sqdist`
  call internally uses no-grad on the bank (the bank is a fixed tensor); the
  subtraction and min over bank still provide gradient to
  `out_minus_all["pred_latent"]`.
- Verify `knn_sqdist` supports autograd through `queries`. If it does not
  (check its implementation), replace with an inline equivalent:
  ```python
  # Inline autograd-safe KNN min-sqdist:
  diff = pred.unsqueeze(1) - bank.unsqueeze(0)   # (B, N_bank, D)
  d2 = (diff * diff).sum(dim=-1)                  # (B, N_bank)
  r_minus_to_pos = d2.min(dim=1).values           # (B,)
  ```
  For large banks, chunk over the bank dimension (like `knn_sqdist` does) and
  take running-min with `torch.minimum`.
  **Recommendation**: write a small autograd-safe helper once, reuse it.
- Return the raw scalar `repel_hinge_val` in the `_run_step` return dict so the
  epoch-level mean flows into `D4Health.repel_hinge_loss`.
- The second forward is an extra cost (~one predictor forward). Acceptable for
  research; if profiling shows a problem, that is a later optimization.

### Where to set `self._current_bootstrap_k`

In `run()`, inside the Phase-B loop, before each epoch's training call
(around `trainer.py:564`), set `self._current_bootstrap_k = k`. For Phase A,
this attribute can stay unset / 0; the guard above handles it.

---

## 8. Logging / wiring

- In `_run_step`'s return dict, add `"repel_hinge": repel_hinge_val`.
- In `_train_one_epoch`'s aggregation (the dict comprehension at
  `trainer.py:378-379`), `repel_hinge` will be aggregated automatically.
- In the Phase-B loop at `trainer.py:571-583`, populate
  `D4Health.repel_hinge_loss=float(train_logs.get("repel_hinge", 0.0))`.
- In the Phase-B `entry` dict at `trainer.py:599-607`, include the repel
  fields so they end up in wandb + stdout + `self.history["phase_B"]`.
- Print the resolved margin and weight at the start of Phase B for auditability.

---

## 9. CLI flags (in `train_d4.py`)

Add after the existing advantage-gate flags (around `train_d4.py:121-133`):

```python
p.add_argument("--repel-weight", type=float, default=0.0,
               help="Phase-B M-step repel loss weight (lambda_rep). 0 disables.")
p.add_argument("--repel-margin", type=float, default=-1.0,
               help="Hinge margin m in L_repel. <0 => auto from d²(fail z_t, B_+) quantile.")
p.add_argument("--repel-margin-percentile", type=float, default=0.5,
               help="Quantile used for auto margin (default 0.5 = median).")
p.add_argument("--repel-on-phase-a", action="store_true",
               help="If set, also apply repel during Phase A (default: Phase B only).")
p.add_argument("--repel-warmup-epochs", type=int, default=0,
               help="Number of Phase-B epochs to skip repel term at the start.")
```

Thread these into `D4TrainerConfig(...)` at `train_d4.py:233-272`.

---

## 10. Shell script (in `scripts/train_d4.sh`)

Add env vars near the existing Phase-B gate block (around `train_d4.sh:102-105`):

```bash
# Phase-B M-step repel loss. Pushes c=- branch AWAY from expert bank B_+
# at every batch element to break f_+ ≡ f_- symmetric fixed point.
# Off by default; recommended first try REPEL_WEIGHT=0.1.
REPEL_WEIGHT="${REPEL_WEIGHT:-0.0}"
REPEL_MARGIN="${REPEL_MARGIN:--1.0}"                     # <0 => auto
REPEL_MARGIN_PERCENTILE="${REPEL_MARGIN_PERCENTILE:-0.5}"
REPEL_ON_PHASE_A="${REPEL_ON_PHASE_A:-0}"
REPEL_WARMUP_EPOCHS="${REPEL_WARMUP_EPOCHS:-0}"
```

Thread to `EXTRA_ARGS`:

```bash
EXTRA_ARGS+=(--repel-weight "${REPEL_WEIGHT}")
EXTRA_ARGS+=(--repel-margin "${REPEL_MARGIN}")
EXTRA_ARGS+=(--repel-margin-percentile "${REPEL_MARGIN_PERCENTILE}")
EXTRA_ARGS+=(--repel-warmup-epochs "${REPEL_WARMUP_EPOCHS}")
if [[ "${REPEL_ON_PHASE_A}" == "1" ]]; then
    EXTRA_ARGS+=(--repel-on-phase-a)
fi
```

---

## 11. Verification checklist (NOT an empirical AUROC test)

These are minimal smoke tests to verify the implementation is plumbed
correctly. **Do not** attempt to verify that collapse is prevented — that is a
separate empirical study the user will run manually.

Required:

1. **Default-off regression**: run `bash scripts/train_d4.sh` with no env
   overrides. Training must behave **byte-identically** (up to nondeterminism)
   to the current `master` behavior: no repel term active, no extra forward,
   no new log fields filled with non-zero values. Phase-A-only
   (`SKIP_BOOTSTRAP=1`) must also be unaffected.

2. **Smoke test with repel on**: run `REPEL_WEIGHT=0.1 bash scripts/train_d4.sh`
   for 1 Phase-A + 2 Phase-B epochs (override `WARM_UP_EPOCHS=1
   BOOTSTRAP_EPOCHS=2 FAIL_NUM=20 SUCC_NUM=20` for speed). Verify:
   - Log contains `[d4_trainer] repel loss active: weight=0.10 margin=<float>`.
   - Per-epoch Phase-B entry contains `repel_hinge` as a finite float
     (≥ 0, may be 0 if hinge is already satisfied everywhere).
   - `D4Health` entries in `health_history` have `repel_hinge_loss` populated.
   - Training does not crash from shape / device / autograd issues.

3. **Checkpoint loadability**: verify the saved `.pt` payload loads cleanly
   with the existing `D4BenchmarkDiscriminator` path (no new required fields
   on the inference side).

4. **`repel_warmup_epochs`**: with `REPEL_WEIGHT=0.1 REPEL_WARMUP_EPOCHS=2
   BOOTSTRAP_EPOCHS=4`, confirm `repel_hinge=0.0` for outer epochs 1–2 and
   nonzero (or potentially zero if hinge is satisfied, but the code path is
   exercised) for 3–4.

---

## 12. Out-of-scope — explicit non-goals for this PR

- No changes to `compute_advantage_gate` or `build_expert_target_bank`.
- No Two-Bank (B_-) construction.
- No `cfg_knn` inference mode. (The `inference/` tree is untouched.)
- No change to `ConditionEmbedder` or `AdaLNDecoderBlock`.
- No empirical tuning — ship with `repel_weight=0.0` default so existing runs
  are unaffected. Tuning is the user's next step after merge.
- No new tests under `tests/` required beyond making sure existing tests pass.

---

## 13. Open implementation decisions (ask if unsure; otherwise use the listed default)

1. **Autograd-safe KNN**: confirm whether `d3disc.filter.knn_sqdist` supports
   gradient through its first argument. If not, implement an inline version in
   `trainer.py` (see §7). **Default**: implement an inline version regardless;
   safer and self-contained.
2. **Repel on the same batch vs. a fresh clean-pos batch**: §7 applies repel
   on the **same** `_run_step` batch (all elements, regardless of `is_fail_raw`
   or sampled `cond`). **Default: yes, same batch.** Do not sample a separate
   batch.
3. **Mean vs sum over batch**: use `.mean()` (as sketched). Keeps the loss
   scale batch-size-invariant; the user tunes `repel_weight`.
4. **`knn_k` for repel**: reuse `cfg.advantage_knn_k` (same bank, same k).
   **Default: yes.**
