# Plan — BCE-WAM Discriminator (density-ratio failure detector on frozen WAM latent)

This document records the **design** of a new failure detector that sits on top of the
existing `lpb_v2` WAM-style pretrained encoder and the **concrete code plan** to add
it to `robosuite/discriminator/lpb_v2/` next to the existing single-bank and two-bank
KNN baselines.

It supersedes `BCE_PROMPT.md` (which is the original idea brief).

---

## 1. Motivation

`lpb_v2` already ships two failure-detection heads on top of the WAM dynamics model:

- `LPBV2KNN` — single-bank KNN on success demos (one-class OOD).
- `TwoBankKNN` — two-bank KNN with both success and failure (GT-sliced) banks.

We have two empirically established facts:

- The pretrained WAM latent at transformer **layer 1** (551-dim) is the most useful
  feature space — the predictor side of WAM is data-starved
  (see `MEMORY.md → project_lpb_v2_predictor_useless`).
- Success vs failure-after-GT classes are **locally separable but not linearly
  separable**; pooled Mahalanobis already hits AUROC ≈ 0.74 with 13pp per-task
  spread (`project_lpb_v2_phase1_separability`). KNN exploits local structure, but
  the parametric Mahalanobis assumption is too rigid.

**Idea.** Instead of modelling `ρ_e(z)` directly, train a binary discriminator
`d(z) = σ(g(z))` so the logit `g(z) ≈ log ρ_e(z) − log ρ_o(z)` becomes a
parametric, learnable **occupancy log-density-ratio**. This is exactly what BCE
gives in the limit, and it fits the geometry of WAM latents better than a
single-Gaussian / linear-LDA assumption.

`g(z)` doubles as an **expert-likeness reward**; `−g(z)` is a **failure cost**.
We use it directly for per-frame failure detection (binary, online).

---

## 2. Problem setting (this project)

| Aspect | Choice |
|---|---|
| Encoder | Frozen WAM, no fine-tuning. Reuse `LPBV2Encoder(feature_source="transformer", transformer_layer=1)`. |
| Input dim | 551 (layer-1 transformer hidden state, flattened). |
| Head input | **Single frame** `z_t`. No temporal window. |
| Sharing | **One head**, trained jointly on all tasks. |
| Output | Scalar logit `g(z)`; failure when `g(z) < τ` (i.e. `−g(z) ≥ τ'`). |
| Task | Per-frame online failure detection on the existing benchmark. |
| Disjoint invariant | Bank/train video_ids **must** be disjoint from benchmark eval video_ids (hard assert), per `feedback_disjoint_fail_pool`. |

GT failure annotation `first_gt_failure_frame` is treated as **absolutely correct**
(user confirmed). No guard band is required for the v2 GT-labeled run.

---

## 3. Two training phases (incremental complexity)

### Phase A — PU-learning baseline (no failure labels)

- `D_e` = all frames from success rollouts.
- `D_o` = all frames from failure rollouts, **treated as negatives without GT slicing**.
  - Note this is biased — failure rollouts contain expert-looking prefix frames.
  - We accept the bias and let the network learn an "average" failure-distribution.
  - This is the classical PU / Nigam-style **labelled positive vs unlabelled-as-negative**
    setup; the resulting `d(z)` is calibrated but biased toward labelling
    near-expert frames as failure. Useful as a **lower bound** that does **not**
    require failure annotations.

### Phase B — GT-labeled BCE (small failure budget)

- `D_e` = success frames **+** failure-trajectory prefix `t < first_gt_failure_frame`.
- `D_o` = failure-trajectory suffix `t ≥ first_gt_failure_frame`.
- Failure-data budget **≤ 10 failure trajectories per task** (small, matching the
  two-bank KNN regime). This is the regime we believe matters in practice.
- No guard band (`first_gt_failure_frame` is GT).

Phase A is run first as a **sanity check** (it should be roughly as bad as KNN
without failure data). Phase B is the **headline method** to compare against the
two-bank KNN.

---

## 4. Model & loss (kept deliberately simple)

### Head

```
g_θ(z) = MLP(z),   z ∈ R^551
```

- Architecture: `[Linear(551, 256) → LayerNorm → GELU → Linear(256, 256) → LayerNorm → GELU → Linear(256, 1)]`
- Output: scalar logit. No sigmoid in the model (we use BCEWithLogitsLoss).
- Regularisation: weight decay `1e-4`, no dropout (head is small and over-regularised
  hurts density-ratio estimation).
- Frozen encoder; head is the only trainable module.

### Loss

`BCEWithLogitsLoss` per frame, with **class-balanced batch sampling** so the prior
correction term is zero. (Alternative: leave unbalanced and apply
`g ← g − log(|D_e|/|D_o|)` at inference — we prefer balanced sampling.)

```
L = E_{z∼D_e}[ softplus(−g_θ(z)) ] + E_{z∼D_o}[ softplus(g_θ(z)) ]
```

No mixup / margin / spectral norm / KNN-distillation in v1 — those are deferred to
§9. We want a clean baseline first.

### Optimisation

- Optimiser: AdamW, lr `3e-4`, weight decay `1e-4`.
- Schedule: cosine, ~20–30 epochs over the cached feature dataset.
- Batch size: 512 frames (256 expert + 256 other via balanced sampler).
- Early stopping on held-out frame BCE / AUROC.

### Calibration

- Disjoint **success calibration split** (re-use `calib_fraction` from
  `LPBV2BenchmarkDiscriminator`).
- Per-task threshold: `τ_task = percentile(g_θ(z) over success-calib frames, δ)`
  i.e. flag the bottom-`δ%` of expert-likeness as failure. **Per-task** because of
  the 13pp inter-task spread.
- Pooled threshold also reported as a sanity comparison.

---

## 5. Evaluation protocol

1. Run on the existing real-world benchmark via `FailureBenchmark` (same loader
   the two-bank KNN uses), with the same train/eval split.
2. Metrics: per-task AUROC, AUPRC, pooled AUROC, 13pp spread reduction.
3. Baselines to plot side-by-side in the result JSON:
   - `LPBV2KNN` (single-bank, layer-1, current best).
   - `TwoBankKNN` (success + failure bank, GT-sliced).
   - Pooled Mahalanobis (Phase 1 baseline, AUROC ≈ 0.74).
4. Both Phase A (PU) and Phase B (GT) are reported.

Headline target: **Phase B AUROC > two-bank KNN AUROC**, with smaller per-task
spread. Phase A is success only if it beats single-bank KNN (which we suspect it
will not — PU labels are too noisy — but the diagnostic value matters).

---

## 6. Disjointness & data handling

Mirror the invariant from `TwoBankBenchmarkDiscriminator._assert_disjoint`:

- Training set: success trajectories (excluding the held-out success-calib split),
  plus failure trajectories from a **disjoint pool**.
- Benchmark eval set: never appears in the training or calibration pools.
- `assert set(train_video_ids) & set(eval_video_ids) == set()` at fit time.

This is enforced by reusing the existing `fail_bank_trajectories` / disjoint-check
plumbing from the two-bank module.

---

## 7. Code modification plan — file-by-file

All changes confined to `robosuite/discriminator/lpb_v2/`.

### 7.1 New module: `bce_discriminator.py`

Mirrors `two_bank_knn.py` in structure. Owns the **head + loss + calibration**
(but not encoding — that stays in `LPBV2Encoder`).

```
class BCEHead(nn.Module):
    def __init__(self, in_dim=551, hidden=256, num_layers=2): ...
    def forward(self, z): -> logit (B,)

@dataclass
class BCECalibStats: ...

class BCEDiscriminator:
    """Train + score head on frozen WAM features.

    fit(expert_features, other_features,
        expert_calib_features, other_calib_features=None,
        epochs=..., lr=..., delta=...)
        - balanced sampling, BCEWithLogitsLoss
        - returns threshold tau

    score(features) -> DetectionResult        # same dataclass as knn.py
        - g_t = head(z_t);  preds = (g_t < tau).long()
    """
```

Reuses `DetectionResult` from `knn.py` so downstream JSON layout is unchanged.

### 7.2 New module: `benchmark_bce.py`

Subclass `LPBV2BenchmarkDiscriminator` exactly the way `benchmark_two_bank.py`
already does. Responsibilities:

- Construct `BCEHead` + `BCEDiscriminator` per benchmark fit.
- Reuse the parent's `_encode(...)` caching and `_assert_disjoint(...)` helper
  (lift it out of `benchmark_two_bank.py` so both adapters can call it, or
  duplicate at first then refactor — duplicate is fine for v1).
- Accept constructor args:
  - `phase: Literal["pu", "gt"]` — selects whether to GT-slice failure
    trajectories or use them whole as `D_o`.
  - `fail_bank_trajectories`, `fail_calib_trajectories` (same naming as two-bank).
  - `max_fail_per_task` (defaults to 10 for `phase="gt"`, unlimited for `"pu"`).
  - Standard head/optim hyperparams: `head_hidden`, `head_layers`, `lr`,
    `epochs`, `batch_size`, `weight_decay`, `delta`, `calib_fraction`, `seed`.
  - `feature_source` defaults to `"transformer"`, `transformer_layer=1`.
- Per-task slicing logic for `phase="gt"`:
  - For each failure trajectory: prefix `[0, first_gt)` → `D_e`,
    suffix `[first_gt, min(T, first_gt+fail_window))` → `D_o`.
  - `fail_window` defaults to `fail_bank_last_k` (60) to match two-bank.
- For `phase="pu"`: entire failure trajectory → `D_o`, no GT label needed.
- After fit, write `calibration_summary()` consistent with the two-bank shape so
  the same downstream analysis scripts work.

### 7.3 New CLI: `run_bce_benchmark.py`

Copy `run_benchmark.py`. Add `--phase {pu,gt}`, `--head-hidden`, `--head-layers`,
`--epochs`, `--lr`, `--weight-decay`, `--batch-size`, `--max-fail-per-task`.
Calls `BCEBenchmarkDiscriminator` instead of `LPBV2BenchmarkDiscriminator`.

For the GT phase, plumb `--fail-bank-* / --fail-calib-*` exactly like
`run_two_bank_real_world_benchmark.py` (look at how two-bank discovers the failure
trajectories from `fail_root`).

### 7.4 New script: `scripts/run_bce_real_world_benchmark.sh`

Bash wrapper mirroring `scripts/run_two_bank_real_world_benchmark.sh`. Defaults:

- `MODEL_CKPT` same Agilex layer-1 checkpoint as the two-bank script.
- `PHASE=gt`, `MAX_FAIL_PER_TASK=10`.
- `KNN_FEATURE_SOURCE=transformer`, `KNN_TRANSFORMER_LAYER=1`.
- `EPOCHS=20`, `LR=3e-4`, `BATCH_SIZE=512`, `HEAD_HIDDEN=256`, `HEAD_LAYERS=2`.
- `DELTA=10`, `CALIB_FRACTION=0.2`.
- Output JSON under `checkpoints/lpb_v2/bce_eval/${RUN_NAME}/benchmark.json`.

### 7.5 `__init__.py`

Export the new public types:

```
from .bce_discriminator import BCEDiscriminator, BCEHead, BCECalibStats
from .benchmark_bce import BCEBenchmarkDiscriminator
```

Append to `__all__`.

### 7.6 Tests

Add `tests/test_bce_discriminator.py`:

- Smoke: random `(N, 551)` expert + other features, fit a 2-layer head for 5
  epochs, assert AUROC on held-out features ≥ 0.9 when the synthetic classes are
  truly Gaussian-separated.
- Disjointness: build trajectories with overlapping `video_id` between eval and
  train pool; assert `BCEBenchmarkDiscriminator.fit_on_benchmark` raises.
- Threshold determinism: with a fixed seed and `delta`, threshold value is
  reproducible across two `fit(...)` calls.

### 7.7 Out of scope for this PR

- KNN/prototype/distillation extensions (§9).
- Temporal windows (single-frame head is enough for the headline number).
- Encoder fine-tuning.

---

## 8. Implementation order

| Step | Deliverable | Est. effort |
|---|---|---|
| 1 | `bce_discriminator.py` (head + fit/score, no benchmark plumbing) | 0.5 day |
| 2 | Unit test on synthetic features (`tests/test_bce_discriminator.py`) | 0.25 day |
| 3 | `benchmark_bce.py` subclassing the v2 adapter; reuse `_encode` cache | 0.5 day |
| 4 | `run_bce_benchmark.py` + shell wrapper | 0.25 day |
| 5 | Phase A run (PU) on real-world benchmark; sanity numbers | 0.25 day |
| 6 | Phase B run (GT, ≤10 fail/task) on real-world benchmark; headline number | 0.5 day |
| 7 | Compare to single-bank KNN, two-bank KNN, Mahalanobis baseline | 0.25 day |

Skip step 5 if step 6 already wins — Phase A is diagnostic only.

---

## 9. Future extensions (explicitly deferred)

Documented here so we don't lose them, but **not** in v1:

- **Temporal windows.** Concatenate `[z_{t-k}, …, z_t]` so the head sees motion.
- **KNN hybrid.** `s(z) = α · (−g(z)) + (1−α) · d_KNN(z, D_e)` with α tuned on
  calibration split.
- **KNN-augmented features.** Append `(min-d to D_e, min-d to D_o, k-th-d, …)`
  to head input.
- **Prototype head.** `g(z) = LSE_j ⟨z, c_e^j⟩ − LSE_j ⟨z, c_o^j⟩`, learnable
  prototypes with diversity loss.
- **Distillation from two-bank KNN.** Add `λ · ‖g(z) − g_KNN(z)‖²` regulariser
  where `g_KNN(z) = log((d_o + ε)/(d_e + ε))`.
- **Calibration variants.** Temperature scaling → probability semantics.
- **Trajectory aggregation.** Beyond per-frame thresholding: cumulative sums,
  changepoint detectors on `g(z)`.

---

## 10. Open invariants & guardrails

- WAM checkpoint, normalizer, and `LPBV2Encoder` config are identical to the
  two-bank KNN run — controlled variable.
- `feature_source="transformer"`, `transformer_layer=1` is the locked feature
  space for v1.
- Disjoint video_id between train/calib and benchmark eval is asserted.
- Threshold is **per-task**, computed on a disjoint success-calib split.
- Phase B failure budget capped at 10 trajectories per task.

---

## 11. References inside this repo

- `knn.py` → `LPBV2Encoder`, `LPBV2KNN`, `DetectionResult`, `knn_min_l2_dist`.
- `benchmark.py` → `LPBV2BenchmarkDiscriminator` (parent for the BCE adapter).
- `two_bank_knn.py` → `TwoBankKNN` (closest sibling for code patterns).
- `benchmark_two_bank.py` → `TwoBankBenchmarkDiscriminator`,
  `_fail_frame_range`, `_assert_disjoint` (reuse / lift).
- `model_loader.py` → `load_model` (do not modify).
- `MEMORY.md`:
  - `project_lpb_v2_phase1_separability` (baseline numbers).
  - `project_lpb_v2_predictor_useless` (why only encoder side).
  - `project_lpb_v2_latent_separability` (locally separable, not linearly).
  - `feedback_disjoint_fail_pool` (hard invariant).
  - `feedback_empirical_over_theoretical` (bias toward shipping simple v1).
