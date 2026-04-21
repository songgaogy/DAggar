# D³-Disc Implementation Plan

> **Audience**: a fresh coding agent that will implement D³-Disc **in-place** on the current `discriminator-baseline` branch.
>
> **Reference spec**: `.claude/discriminator/sfv_discriminator_design.md` (read the full spec first; decisions locked in Appendix C).
>
> **Reference context**: `.claude/discriminator/conversation_context.md`.
>
> **User preferences**: follow `AGENTS.md` strictly (minimal modifications, English code comments, concise, WandB offline for logging, `daggar` conda env).

---

## 1. High-level summary

Build a new discriminator `D3BenchmarkDiscriminator` that plugs into the existing `data.utils.benchmark` framework exactly like `LPBBenchmarkDiscriminator`, but uses:

1. **Flow-Multi pretrained policy encoder** (`robosuite/policy/flow_multi/model.py`) as the feature backbone, with cache reuse from `data/.lpb_score_cache/<task>/<sha1>.npz`.
2. **Two pooled multi-task KNN banks**: $\hat p_+$ (success) and $\hat p_-$ (cleaned fail).
3. **F3 static self-consistent soft-weight filter**: `w_j^- = σ(β [d_knn²(φ_j, D_+) − κ])`.
4. **Weighted 1-NN KDE** for $\hat p_-$ (soft weights folded as additive log penalties into distances).
5. **Dichotomous step-wise score** `λ_t = (1+ω) · d²(φ_t, D_+) − ω · d²_weighted(φ_t, D_-)`.
6. **Per-task conformal calibration** from held-out success frames.

Phase 1 = step-wise only (K=0). No dynamics rollout.

---

## 2. File/module layout

Create a new subpackage under `robosuite/discriminator/`:

```
robosuite/discriminator/d3disc/
├── __init__.py
├── encoder.py             # FlowMultiEncoderWrapper + cache reader
├── filter.py              # F3 soft-weight + weighted KNN helpers
├── detector.py            # D3Detector (pooled bank, fit, score)
├── d3_benchmark.py        # D3BenchmarkDiscriminator (benchmark API)
├── visualize.py           # analogous to lpb/visualize.py
├── config/
│   └── d3_detector.yaml   # hydra config (if used)
└── scripts/
    ├── run_d3_benchmark.sh
    ├── visualize_d3.sh
    └── ablate_omega_beta_k.sh
```

**Do not modify** existing `robosuite/discriminator/lpb/` — D³-Disc is a sibling discriminator, not a replacement. `LPBBenchmarkDiscriminator` remains as the ω=0 baseline.

---

## 3. Dependencies and imports

- `robosuite.policy.flow_multi.model.build_flow_policy` (or MultiModalFlowPolicy init) for encoder
- `robosuite.policy.flow_multi.eval_flow.resolve_language_instruction` for task prompt
- `robosuite.policy.flow_multi.utils.env_util.RobosuiteProprioExtractor` for proprio
- `data.utils.benchmark.BenchmarkTrajectory`, `DiscriminatorOutput`, `FailureBenchmark`
- `h5py`, `numpy`, `torch`, `hashlib`

---

## 4. Component specs

### 4.1 `encoder.py` — `FlowMultiEncoderWrapper`

**Purpose**: load flow_multi policy checkpoint → produce per-frame latent `z_t ∈ ℝ^256`. Reuses existing cache if present; computes and saves on cache miss.

**Constructor**:
```python
class FlowMultiEncoderWrapper:
    def __init__(
        self,
        policy_ckpt_path: str,          # e.g., checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_*.pt
        cache_root: str = "data/.lpb_score_cache",
        device: str = "cuda",
        image_size: int = 128,
        encoder_batch_size: int = 128,
    ): ...
```

**Cache key** (must match existing cache convention to enable reuse):
```python
def cache_key(self, task_name: str, file_path: str, demo_key: str) -> str:
    key = "|".join([
        str(task_name),
        str(Path(file_path).resolve()),
        str(demo_key),
        str(self.image_size),
        ",".join(self.camera_names),
        str(os.path.getmtime(file_path)),
        str(Path(self.ckpt_path).resolve()),
    ])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()
```
Cache file path: `<cache_root>/<task_name>/<sha1>.npz` with keys `latents`, `actions`, `task_name`, `file_path`, `demo_key`.

**Key methods**:
- `load_or_encode(task_name, file_path, demo_key) -> EncodedDemo`  
  Return `EncodedDemo(latents=(T,256), actions=(T,7), task_name, file_path, demo_key)`.
- `_encode_demo(...)`: **only called on cache miss.** Build inputs identical to the pipeline that populated the cache (images (T,V,3,128,128) normalized with ImageNet mean/std; proprio via `RobosuiteProprioExtractor`; language resolved from task prompt map).

**Reference implementation**: the git-historical `robosuite/discriminator/lpb_score/core/policy_encoder.py` (available via `git show 5056929:robosuite/discriminator/lpb_score/core/policy_encoder.py`). Copy the encoder setup but strip LoRA, training, preprocessed-bundle, and unrelated code — only keep:
- checkpoint loading + `build_flow_policy`
- `encode_context_tensors` / `encode_sequence`
- `cache_key` / `cache_path` / `load_or_encode_demo` / `save_encoded_demo`
- image resize + ImageNet mean/std normalization
- proprio extraction + prop_mean/prop_std normalization

**Verification**: after implementation, load one existing cache file and compare to a fresh encode of the same demo — should match bit-exact (deterministic forward).

### 4.2 `filter.py` — F3 soft weight + weighted KNN

```python
def compute_f3_weights(
    fail_feats: torch.Tensor,           # (N_-, D)
    pos_bank: torch.Tensor,             # (N_+, D)
    beta: float | None = None,          # if None: use 1/MAD auto
    kappa: float | None = None,         # if None: use self-KNN median auto
    knn_chunk_size: int = 8192,
    k: int = 1,
) -> tuple[torch.Tensor, float, float]:
    """
    Returns (w_minus, beta_used, kappa_used).
    """
    ...
```

**Implementation details**:
- Compute `d² = knn_avg_sqdist(fail_feats, pos_bank, k=k)` — 1-NN default, generalize to kNN avg.
- `kappa_used`: if `kappa is None`, compute `self_d² = knn_avg_sqdist(pos_bank, pos_bank, k=k, exclude_self=True)`; take `kappa = median(self_d²)`.
- `beta_used`: if `beta is None`, compute `MAD(self_d²) = median(|self_d² − median|)`; `beta = 1 / max(MAD, eps)`.
- `w = sigmoid(beta * (d² - kappa))`.

```python
def weighted_knn_sqdist(
    query: torch.Tensor,               # (B, D)
    bank: torch.Tensor,                # (N, D)
    bank_weights: torch.Tensor,        # (N,) in (0, 1]
    sigma_sq: float = 0.5,             # 2σ² = 1 when sigma=1/√2
    chunk_size: int = 8192,
    k: int = 1,
) -> torch.Tensor:
    """
    Weighted 1-NN (k-NN) distance with soft weights folded as additive
    log-penalties: effective_dist² = ||q − bank_j||² − 2σ² log(w_j).
    Low-weight bank points are pushed "further" in effective distance.
    """
    ...
```
Use `torch.cdist` chunked, add `−2*sigma_sq*log(weights)` per bank point before taking min (or top-k mean).

### 4.3 `detector.py` — `D3Detector`

Analogous to `AdaptiveKNNDiscriminator` in `lpb/knn_discriminator.py`, but with two banks:

```python
class D3Detector:
    def __init__(
        self,
        omega: float = 0.5,
        k: int = 1,
        beta: float | None = None,      # None -> auto
        kappa: float | None = None,     # None -> auto
        delta: float = 10.0,
        knn_chunk_size: int = 8192,
        device: str = "cuda",
        sigma_sq: float = 0.5,
    ): ...

    def fit(
        self,
        pos_sequences: list[torch.Tensor],         # per-traj success features
        neg_sequences: list[torch.Tensor],         # per-traj fail features (before F3)
        calibration_sequences: list[torch.Tensor], # per-task held-out success
    ) -> float:
        """
        1. Concat pos_sequences -> self.pos_bank (N_+, D).
        2. Concat neg_sequences -> self.neg_bank (N_-, D).
        3. Compute F3 weights w_minus for neg_bank (using pos_bank alone).
        4. For each calibration sequence:
             d_pos² = knn(φ, pos_bank, exclude-self if applicable)
             d_neg² = weighted_knn(φ, neg_bank, w_minus)
             λ = (1+ω) * d_pos² − ω * d_neg²
        5. tau = percentile(λ_calib_all, q=1-delta/100).
        Return tau.
        """
        ...

    @torch.no_grad()
    def score(
        self,
        features: torch.Tensor,       # (T, D)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (lambdas, preds, thresholds_per_step) for a single sequence.
        thresholds_per_step = constant self.tau repeated (matches LPB API).
        """
        ...
```

**Important**: when `omega == 0`, skip the $\hat p_-$ branch entirely (save compute; exact LPB-equivalent path). Assert `neg_sequences` may be empty in that case.

### 4.4 `d3_benchmark.py` — `D3BenchmarkDiscriminator`

Mirror `LPBBenchmarkDiscriminator` API exactly (users of the benchmark framework won't need to change anything).

```python
class D3BenchmarkDiscriminator:
    name = "d3_disc"

    def __init__(
        self,
        *,
        policy_ckpt_path: str,
        cache_root: str = "data/.lpb_score_cache",
        device: str = "cuda",
        # D³ hyperparameters
        omega: float = 0.5,
        k: int = 1,
        beta: float | None = None,
        kappa: float | None = None,
        delta: float = 10.0,
        knn_chunk_size: int = 8192,
        # calibration / data
        calib_fraction: float = 0.2,
        seed: int = 0,
        # task scope
        share_banks_across_tasks: bool = True,   # D8 default
        per_task_calibration: bool = True,
        verbose_fit: bool = True,
    ): ...

    def fit_on_benchmark(self, trajectories: list[BenchmarkTrajectory]) -> None:
        """
        1. Partition trajectories by task, then by is_failure.
        2. For each success trajectory: encode (via cache) -> pos_features per task.
        3. For each fail trajectory: encode -> neg_features per task.
        4. If share_banks_across_tasks=True (D8 default):
             pool all pos_features -> single pos_bank
             pool all neg_features -> single neg_bank
           Else:
             per-task banks (fall back to original LPB structure).
        5. For each task:
             split per-task success pool into bank_trajs + calib_trajs (using calib_fraction).
             Actually: since banks are shared, just use per-task calibration split from the shared pool.
           Build D3Detector, fit(), store detector per task (or one shared).
        6. Per-task threshold τ stored in self._tau_per_task.
        """
        ...

    def score_trajectory(self, traj: BenchmarkTrajectory) -> DiscriminatorOutput:
        """
        Encode traj -> features -> detector.score -> DiscriminatorOutput.
        """
        ...

    def calibration_summary(self) -> dict: ...
    def close(self) -> None: ...
```

**Calibration split implementation detail**: even with shared banks, calibration must be per-task-exchangeable. Simplest implementation:
- `self._pos_bank`: pooled features from **all** success trajectories **not** in any task's calibration split.
- `self._calib_features[task]`: features from that task's calibration slice.
- Score each calib slice through the *same* shared bank and fit τ from the resulting λs.

### 4.5 `visualize.py`

Copy-adapt `lpb/visualize.py`:
- Replace `LPBBenchmarkDiscriminator` construction with `D3BenchmarkDiscriminator`.
- Add HUD line showing both `d_pos²` and `d_neg²` per frame, plus `w_minus` if frame is from a fail traj (diagnostic only).
- Output same PDF + mp4 structure.

Keep existing `lpb/visualize.py` untouched.

---

## 5. CLI entry points

### 5.1 `scripts/run_d3_benchmark.sh`

```bash
#!/usr/bin/env bash
# D³-Disc benchmark driver; mirrors lpb_knn benchmark script layout.
set -euo pipefail

PYTHON=/home/dodo/miniconda3/envs/daggar/bin/python
POLICY_CKPT=${POLICY_CKPT:-checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt}
CACHE_ROOT=${CACHE_ROOT:-data/.lpb_score_cache}
FAIL_ROOT=${FAIL_ROOT:-data}       # <task>/fail_rollout/*.hdf5 pattern
SUCCESS_ROOT=${SUCCESS_ROOT:-data} # <task>/success_rollout/*.hdf5

OMEGA=${OMEGA:-0.5}
KNN_K=${KNN_K:-1}
BETA=${BETA:-auto}
DELTA=${DELTA:-10.0}

$PYTHON -m robosuite.discriminator.d3disc.run_benchmark \
    --policy-ckpt "$POLICY_CKPT" \
    --cache-root "$CACHE_ROOT" \
    --fail-root "$FAIL_ROOT" \
    --success-root "$SUCCESS_ROOT" \
    --omega "$OMEGA" \
    --k "$KNN_K" \
    --beta "$BETA" \
    --delta "$DELTA" \
    "$@"
```

### 5.2 `scripts/ablate_omega_beta_k.sh`

Sweep the ablation matrix per D1 + D5:
```bash
for OMEGA in 0 0.1 0.2 0.5 1; do
  for K in 1 5; do
    for BETA_MUL in 0.5 1 2 4; do
      OMEGA=$OMEGA KNN_K=$K BETA=auto_x$BETA_MUL bash scripts/run_d3_benchmark.sh \
          --out-suffix "w${OMEGA}_k${K}_b${BETA_MUL}"
    done
  done
done
```

### 5.3 `scripts/visualize_d3.sh`

Analogous to `lpb/scripts/visualize_lpb.sh`, passes `--d3-ckpt`-equivalent args (actually policy ckpt + hyperparams; no trained model).

---

## 6. Concrete task breakdown

### Phase A — Plumbing (no algorithmic work)
1. Create the subpackage skeleton `robosuite/discriminator/d3disc/` with `__init__.py` exposing public classes.
2. Implement `FlowMultiEncoderWrapper` in `encoder.py`. Verify against existing cache:
   - Load one `.npz` from `data/.lpb_score_cache/PickPlaceBread/`, record its `(file_path, demo_key)`.
   - Re-encode with fresh wrapper; compare `latents` bit-exact (assert max abs diff < 1e-5).
3. Implement `compute_f3_weights` and `weighted_knn_sqdist` in `filter.py`. Unit-test:
   - On fake data, verify weight ∈ (0,1); dense region → small w; sparse region → large w.
   - Weighted KNN: setting all weights to 1 should match unweighted KNN.

### Phase B — Core discriminator
4. Implement `D3Detector` in `detector.py`. Structural mirror of `AdaptiveKNNDiscriminator`.
5. Implement `D3BenchmarkDiscriminator` in `d3_benchmark.py`. Pattern-match against `LPBBenchmarkDiscriminator.fit_on_benchmark` for data partitioning.
6. Top-level driver module `run_benchmark.py` that constructs `D3BenchmarkDiscriminator`, calls `fit_on_benchmark`, iterates `score_trajectory` over test trajectories, dumps metrics JSON (accuracy, F1, AUC, first-failure-delay per task).

### Phase C — Validation & ablation
7. Sanity-check: with `omega=0`, expect metrics equal to current LPB baseline (within conformal variance).
8. Sweep ablation per §5.2; aggregate into a single CSV + WandB offline run.
9. Visualization: pick 4 fail trajectories per task, run `visualize_d3.sh`, inspect PDFs for α=0 vs α=0.5 vs α=1.

### Phase D — Docs
10. Update `.claude/discriminator/sfv_discriminator_design.md` if any discrepancy discovered during implementation.
11. Add a brief `README.md` under `robosuite/discriminator/d3disc/` summarizing how to run + pointing at the design doc.

---

## 7. Things NOT to do

- Do **not** retrain the flow_multi policy. Use only `checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_*.pt`.
- Do **not** retrain the LPB dynamics model. D³-Disc uses only the flow encoder; LPB's ResNet-18 + predictor is irrelevant here.
- Do **not** modify `robosuite/discriminator/lpb/` — new code lives in sibling package.
- Do **not** introduce new dependencies. Stick to torch, numpy, h5py, hydra already used by LPB.
- Do **not** implement K-step rollout yet (D2). Phase 1 is K=0.
- Do **not** implement iterative F3-EM (D11). Static F3 only.
- Do **not** implement F1/F2 filters (D3). F3 only.
- Do **not** try to clean the cache or regenerate it; reuse as-is.

---

## 8. Deliverables checklist

- [ ] `robosuite/discriminator/d3disc/` package with all modules above.
- [ ] Cache-reuse works: encoding an existing-cached demo loads from disk, encoding a new demo writes to cache with the right sha1 hash.
- [ ] `D3BenchmarkDiscriminator` passes the same benchmark interface tests as `LPBBenchmarkDiscriminator`.
- [ ] With `omega=0`, results numerically match current LPB baseline on at least one task.
- [ ] Ablation CSV + visualizations for `ω × k × β` sweep.
- [ ] Bash scripts under `scripts/` (per AGENTS.md: "if one code script requires long CLI parameters, please write `.bash` running entrance").
- [ ] WandB offline run (`WANDB_MODE=offline`, `WANDB_NAME=songgao-personal`).
- [ ] Per-task metrics table + overall multi-task detector metrics.

---

## 9. Open questions to raise during implementation (only if blocked)

These should only be raised if you hit something ambiguous in the spec. Do not re-ask user before starting.

1. Should the action be part of the feature $\varphi$ or only the visual-proprio latent $z$? **Default**: concatenate normalized action to $z$ (matches LPB's treatment). Only revisit if empirically bad.
2. When a fail trajectory spans many `(s,a)` transitions, should we subsample to match success bank size? **Default**: no subsampling; soft weights already downweight expert-like frames.
3. For `PandaLift` (no fail_rollout), should it contribute to the shared $\hat p_-$? **Default**: it can't — skip cleanly (only tasks with ≥ 1 fail trajectory contribute to $\mathcal{D}_-$).

---

## 10. Final check before merging

Run, in order, from repo root with `daggar` env activated:
```bash
WANDB_MODE=offline WANDB_NAME=songgao-personal \
  bash robosuite/discriminator/d3disc/scripts/run_d3_benchmark.sh \
  --omega 0
# expect metrics ≈ current LPB baseline

WANDB_MODE=offline WANDB_NAME=songgao-personal \
  bash robosuite/discriminator/d3disc/scripts/run_d3_benchmark.sh \
  --omega 0.5
# expect metrics ≥ LPB baseline (or at least not catastrophically worse on PandaLift)

bash robosuite/discriminator/d3disc/scripts/ablate_omega_beta_k.sh
# full sweep + CSV

bash robosuite/discriminator/d3disc/scripts/visualize_d3.sh --task PickPlaceBread --num-trajs 4
# diagnostic PDFs
```

Report final metrics table + Chinese summary per `AGENTS.md` transparency rule.
