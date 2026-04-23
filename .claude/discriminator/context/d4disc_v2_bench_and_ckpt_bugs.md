# D4-Disc v2 — Benchmark & Checkpoint Bugs (post-fix-landing investigation)

> Context: after landing the v1 post-mortem fixes (rank warm-start, Phase A
> clean-only, skip-bootstrap mode) the fail/success discrimination on the
> benchmark was still wrong. Debugging found one real benchmark bug (fixed)
> and one unresolved deeper issue in the trained predictor itself.
> Source runs:
> - Training: `checkpoints/d4disc/dynamics/d4dyn_20260422_063722/` (Phase A
>   only, 20 epochs, F2+F1 landed, `skip_bootstrap=0` but only Phase A
>   epochs were saved — the ckpts we tested are `Aep0010`, `Aep0020`).
> - Benchmark (inverted): `checkpoints/d4disc/eval/run_20260422_064143/`.
> - Benchmark (`LAMBDA_MODE=max`, same inversion): `run_20260422_065247/`.
> - Benchmark (after action-horizon fix): `run_fixed_070251/`.
> - Related algorithm-design context: `d4disc_v2_algorithm_design.md`.

---

## 1. Observation that started the debug

With `OMEGA=0` (pos-branch-only baseline via `skip_bootstrap=1`), benchmark
scored **below random**:

| metric | pooled | PickPlaceBread | Can | Cereal | Milk |
|---|---|---|---|---|---|
| traj AUROC | **0.442** | 0.256 | 0.208 | 0.262 | 0.215 |
| frame AUROC | **0.739** | 0.648 | 0.785 | 0.821 | 0.559 |

`AUROC(-score)` jumped to **0.74–0.79** per task — a clean sign inversion.

Key smell: **frame AUROC is strong (0.74) but trajectory AUROC is inverted**.
Per-task scores (mean of `trajectory_score = max(λ_t)`):

```
PickPlaceBread : SUCC 94.3  FAIL 82.8   (FAIL < SUCC)
PickPlaceCan   : SUCC 142.4 FAIL 113.3
PickPlaceCereal: SUCC 184.5 FAIL 159.6
PickPlaceMilk  : SUCC 66.5  FAIL 52.0
```

Success trajectories consistently scored **higher** than fail. All 4 tasks.

---

## 2. First red herring (ruled out)

Hypothesized the `_aggregate_lambda(mode="mean", window=-1)` prefix-cumulative
mean (in `d3disc/detector.py`) was amplifying single-frame spikes with a
small denominator. Changing `LAMBDA_MODE=max` did **not** fix it:

| metric | pooled (max) | per-task |
|---|---|---|
| traj AUROC | 0.460 | 0.34 / 0.19 / 0.44 / 0.20 |
| frame AUROC | 0.832 | 0.70 / 0.84 / 0.91 / 0.80 |

Same inversion. Same magnitude. The aggregator is NOT the problem.

Separate note (still mildly suspect, not the root cause): `d4_benchmark.py:270`
feeds `result.lambda_values` (= `_aggregate_lambda(...)`) as `step_scores` to
the benchmark's `aggregate_trajectory_score`. That's indirected: `step_scores`
is really a prefix-transformed signal. D3's adapter does the same thing and
works fine, because D3's raw per-frame signal is strong enough to survive the
transform. Worth cleaning up eventually, but not what's blocking D4.

---

## 3. The real benchmark bug — action-horizon mismatch (FIXED)

### 3.1 Discovery

Compared training dataset's output for `(file, demo, t=0)` to benchmark's
extracted `D4Frames` for the same trajectory. All inputs byte-identical:

| field | max\|train − bench\| |
|---|---|
| `z_current` | 0.000000 |
| `proprio` | 0.000000 |
| `action` (first action only) | 0.000000 |
| `z_target` | 0.000000 |

But the **shapes** differed:

| field | training (`dataset.py`) | benchmark (`dynamics_feature.py`) |
|---|---|---|
| action sequence | `(B, 1, A)` — `actions[t:t+horizon]`, horizon=1 | `(B, 32, A)` — `actions[t:t+action_horizon]` |

`action_horizon = max_action_horizon = 32` from `arch_args`. The training
horizon is `1`. Benchmark was feeding the predictor 32-action chunks.

### 3.2 Why this breaks inference

Predictor forward (`model.py::ConditionalDynamicsPredictor.forward`) builds
a transformer sequence:

```
x = cat([obs, prop, act, pz, ps], dim=1)    # (B, 1+1+H+1+1, d)
x = x + pos_embedding[:, :seq_len, :]
```

- Training: seq_len = 1+1+1+1+1 = **5**. pred_latent at position 3, pred_proprio at 4.
- Benchmark: seq_len = 1+1+32+1+1 = **36**. pred_latent at position 34, pred_proprio at 35.

The trained positional embedding for pred tokens was at (3, 4). At benchmark
it's read at (34, 35) — totally different learned positions. The causal
attention mask is also different shape. Output is essentially garbage.

### 3.3 Fix

`robosuite/discriminator/d4disc/dynamics_feature.py`:
- `_action_chunks(actions, t_len, horizon)` — new `horizon` parameter; chunk
  length = `horizon` (not `self.action_horizon`).
- Validates `1 <= horizon <= self.action_horizon`.
- `extract(..., horizon)` threads its `horizon` arg through.

Post-fix `D4Frames.action_chunks.shape = (T, 1, A)` — matches training.

### 3.4 Impact

| metric | pre-fix (max mode) | post-fix (max mode) |
|---|---|---|
| traj AUROC pooled | 0.460 | 0.428 (still below 0.5, see §4) |
| frame AUROC pooled | 0.832 | **0.798** |
| per-task traj AUROC | 0.34 / 0.19 / 0.44 / 0.20 | 0.45 / 0.19 / 0.35 / **0.60** (Milk jumps) |
| per-task frame AUROC | 0.70 / 0.84 / 0.91 / 0.80 | 0.62 / 0.83 / 0.91 / 0.75 |
| mean residual on success | 80–200 | **30–60** |

Frame AUROC is still strong (0.80 pooled). Residuals dropped by ~2× into a
more physically meaningful range. Trajectory AUROC still mixed because of
issue §4 — not because of this bug.

**The fix itself is correct and should be kept regardless of what else is
happening.** Train-vs-bench input shape must agree.

---

## 4. The unresolved deeper issue — training-log vs ckpt-weight mismatch

With action-horizon fixed AND cached latents verified byte-identical,
direct forward of the `Aep0020` ckpt on 3990 pooled success-trajectory
frames (EMA weights, `.eval()`) gives:

| metric | training log @ Aep0020 | actual ckpt forward |
|---|---|---|
| `F.mse_loss(pred, tgt)` (per-element) | **0.005** | **0.155** |
| per-sample `r_plus = ‖pred − tgt‖²` | **0.998** | **39.7** |

**30× discrepancy.** The log says the model is well-trained; the weights it
saved say it isn't.

Further:

| ckpt | per-elem MSE (EMA) | per-elem MSE (raw) |
|---|---|---|
| `Aep0010` | **0.117** | 0.135 |
| `Aep0020` | **0.155** | 0.166 |
| `final` (identical to Aep0020) | 0.155 | 0.166 |

Model got **worse** from epoch 10 to epoch 20 under eval — while training log
showed `latent_mse` monotonically dropping 0.034 → 0.005 across those epochs.

### 4.1 Baseline comparison makes this concrete

Tested `f(z, a) ≡ z` (identity predictor — no learning, just copy input) on
the same data:

| predictor | per-elem MSE |
|---|---|
| random init | 1.335 |
| **identity `f(z,a)=z`** | **0.010** |
| trained Aep0020 (EMA) | 0.155 |

**The trained model is 15× WORSE than the trivial identity baseline.**
Latents barely change between consecutive frames (`‖z[t+1] − z[t]‖² ≈ 0.01`
per-element), so any reasonable dynamics predictor should achieve at most
that. Aep0020 achieves 0.155.

### 4.2 What the model did learn

- Per-dim `corr(pred, tgt) = 0.942` — direction right, relative ranking right.
- `std(pred) = 0.83` vs `std(tgt) = 1.02` — output is **under-scaled by ~20%**.
- Best linear rescale `pred · 1.143 − 0.017 → tgt` drops MSE marginally
  (0.155 → 0.141), so the gap is genuine noise, not just a scale bug.

Residual noise std per element ≈ 0.32. Identity baseline would have noise
std ≈ 0.07. Trained model adds 4× more noise than doing nothing.

### 4.3 What we ruled out

- **Input mismatch**: verified byte-identical `(z, proprio, action, target)`
  between training dataset `__getitem__` and benchmark `D4FeatureExtractor.extract`
  for the same `(file, demo, t)`.
- **Wrong weights loaded**: `strict=True` load passes. EMA ≠ raw (EMA diff
  from raw is non-trivial per-param), so EMA shadow was updating. Weight
  magnitudes look reasonable (not all zero, not NaN).
- **Eval vs train mode**: both give similar MSE (0.155 eval, 0.161 train).
  Dropout isn't saving us 30×.
- **Encoder cache mismatch**: `cache_key` is deterministic in
  `(task, resolve(file_path), demo, image_size, cameras, file_mtime, policy_ckpt)`.
  All components match between training and our debug. Source hdf5 mtime
  hasn't changed since before training. Cached latents are reusable.
- **BF16 autocast**: `use_bfloat16=False` in ckpt config; trainer uses
  `nullcontext()`.

### 4.4 What's plausibly the cause (unverified hypotheses)

In rough order of likelihood:

1. **The training `latent_mse` number in the log is wrong**, i.e. computed
   on a different metric than what the comment claims. The code path
   (`_run_step` → `F.mse_loss(pred_latent, target_latent)` with default
   mean reduction) *looks* right, but given that (a) the log number is ~30×
   tighter than what the weights produce, and (b) the log shows monotone
   decrease while eval MSE increases, something in the reporting pipeline
   is not measuring what we think.

2. **Target preprocessing silently applied somewhere**. If training data has
   targets scaled by some factor that `D4FeatureExtractor` doesn't replicate,
   loss is computed on a different distribution than benchmark. But we've
   diffed the dataset outputs byte-identically, so this would have to be
   INSIDE `_run_step` or via `_dtype_context`.

3. **Phase A training setup has a numerical issue**. The under-scaled
   prediction (`std(pred)/std(tgt) = 0.81`) smells like the final affine
   / `latent_head` is sitting at initialization for some dimensions, and
   the trained weight is only recovering ~80% of the target scale. But
   correlation is 0.94, so it's not a total collapse — more like the head
   is under-fitted and the "effective" trained model is weaker than logged.

4. **Post-merge contamination**: after F2 (Phase A routes fail_raw to null
   only) we switched to a clean-only loader and routed everything to c=+
   in Phase A. The `cond_null` logged is 0, which is correct. But maybe the
   model's c=∅ branch is never trained, and the inference uses c=+ — which
   was trained but apparently poorly. This doesn't explain the log-vs-eval
   mismatch either.

None of these is verified. **The user said ckpts can be easily re-trained,
so we stopped chasing this in conversation** and recommended a small sanity
test (below) before the next training run.

---

## 5. Recommended next steps

### 5.1 Immediate: sanity-instrument the next training run

Add a one-time eval block inside Phase A `run()` that computes `r_plus` on
the same few held-out frames **under both train() and eval() modes**, and
logs `(pred_std, tgt_std, corr, identity_baseline_mse)` alongside
`latent_mse`. If the new run shows the same log ≫ eval gap, the log is
mis-measuring. If it matches, the prior ckpt was corrupted somehow.

### 5.2 Verify the predictor beats identity

Any well-trained dynamics predictor on this data must beat
`f(z,a) = z` (MSE ≤ 0.01 per element). Gate training acceptance on this.
Trivially achievable by initializing `pred_head` as near-identity (e.g.
a residual connection from z_current to pred_latent with small multiplier).

### 5.3 Only after §5.1 + §5.2 pass: re-test benchmark

With action-horizon fix already in, and with a predictor that genuinely
outperforms identity, the frame-level AUROC should carry over to
trajectory-level AUROC (via any reasonable aggregator). If it still
inverts, there's a deeper benchmark bug; if it works, we're done on the
benchmark side.

### 5.4 Skip-bootstrap mode is still the right first milestone

The plan from the previous conversation still stands: run with
`SKIP_BOOTSTRAP=1`, `OMEGA=0` for the pos-branch-only baseline. But the
predictor needs to actually be trained first.

---

## 6. Files touched in this investigation

| file | change |
|---|---|
| `robosuite/discriminator/d4disc/dynamics_feature.py` | `_action_chunks` takes `horizon` param; `extract` threads `horizon` through; checks `1 <= horizon <= action_horizon`. |

No changes to predictor / trainer / dataset. The log-vs-eval mismatch (§4)
was not fixed — it was identified and deferred to the re-training run.

---

## 7. Runtime/data artifacts for future reference

- Cached latents live in `data/.lpb_score_cache/<task>/<sha1>.npz`.
  `cache_key = sha1(task | resolve(file_path) | demo_key | image_size |
  cameras | file_mtime | policy_ckpt)`.
- `data/utils/fail_rollout/<task>/out.hdf5` (labeled fail) has `source_hdf5_path`
  + `source_demo_key` attrs pointing to `/home/dodo/data1/robosuite/raw/...`
  (symlink/copy of the training hdf5 under `data/<task>/fail_rollout/`).
  States + actions in labeled hdf5 are byte-identical to source.
- Benchmark uses `source_hdf5_path` for latent encoder lookup (cache hit)
  and `file_path` (labeled hdf5) for `load_states()` / `load_actions()`.
  These are consistent iff the labeled hdf5 preserves source states/actions
  verbatim — which it does.
- Trajectory length is uniformly 400 for both success and fail (no early
  termination; fail trajectories continue past `first_gt_failure_frame`
  with the policy flailing in the failure state).
