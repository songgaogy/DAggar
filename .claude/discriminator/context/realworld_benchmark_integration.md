# Real-world (Agilex) Benchmark Integration Guide

How to plug a **new discriminator** into the real-world Agilex failure-detection
benchmark. Pulled from the logpZO / LPB integration experience; covers the
shared-core API, dataset layout, the entry-script template, EvalConfig knobs,
and the non-obvious pitfalls.

---

## 1. Architecture overview

The failure-detection benchmark has a **source-agnostic core** plus
**source-specific skins** (real-world Agilex, robosuite sim). New
discriminators live under `robosuite/discriminator/<name>/` and are wired into
the benchmark via a single `score_trajectory` call.

```
benchmark/
├── core/
│   ├── trajectory.py      BenchmarkTrajectory base (lazy load_* methods)
│   ├── discriminator.py   Discriminator Protocol + DiscriminatorOutput dataclass
│   ├── benchmark.py       FailureBenchmark.evaluate() + EvalConfig + BenchmarkResult
│   └── metrics.py         AUROC / AUPRC / event / detection-delay metrics
├── real_world/            (Agilex skin)
│   ├── trajectory.py      AgilexBenchmarkTrajectory (hdf5 + .npy cache)
│   ├── loader.py          discover_agilex_trajectories / discover_cached_*
│   ├── __init__.py        FailureBenchmark wrapper (constructor)
│   ├── examples/
│   │   ├── run_dummy.py     end-to-end smoke test (no encoder)
│   │   ├── run_logpZO.py    real discriminator entry (template)
│   │   └── build_cache.py   one-time HDF5 -> .npy raw cache builder
│   └── scripts/
│       ├── build_cache.sh
│       └── run_benchmark.sh (per-discriminator entry; copy + rename)
└── robosuite/             (sim skin; same shape, different loader)
```

Public imports for any new discriminator:

```python
from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput, EvalConfig
from benchmark.real_world import FailureBenchmark
```

---

## 2. Discriminator interface contract

A discriminator only needs **one method** (`score_trajectory`) plus an
optional `fit_on_benchmark` if it needs to train / calibrate.

```python
class MyDiscriminator:
    name: str = "my_method"

    # Optional: pre-evaluation training / calibration over success trajectories.
    # Called once before evaluate(); receives ALL trajectories (failure + success).
    def fit_on_benchmark(self, trajectories: list[BenchmarkTrajectory]) -> None: ...

    # Required: score one trajectory.
    def score_trajectory(self, traj: BenchmarkTrajectory) -> DiscriminatorOutput: ...

    # Optional: cleanup hook (close encoder, free GPU mem, etc.).
    def close(self) -> None: ...
```

### `DiscriminatorOutput` (must match `traj.num_frames` exactly)

| Field | Type | Required | Notes |
|---|---|---|---|
| `step_scores` | `(T,) float` | ✅ | Higher value ⇒ more failure-like. Used for AUROC/AUPRC + trajectory aggregation. |
| `predictions` | `(T,) {0,1} int` | ⚠ | Used **only** when `EvalConfig.step_binarize_strategy="provided"`. |
| `first_failure_frame` | `int \| None` | ⚠ | If absent, derived from `predictions`. Recorded in per-trajectory output. |
| `aux` | `dict` | optional | Free-form (threshold info, hyper-params). Persisted in nothing — useful only if you read it back. |

The benchmark calls `output.validate(expected_T=traj.num_frames)`; mismatched
length is a hard error.

### Key rules

- **Score sign**: higher = more anomalous / more failure-like. The benchmark
  does not invert; if your method outputs negative log-likelihood, that's
  already the right convention. If you output likelihood, negate it.
- **Score length must equal `traj.num_frames`**. The Agilex loader takes
  `min(action.shape[0], qpos.shape[0], image.shape[0], failure_mask.shape[0])`,
  so trim your features to that length, not the raw HDF5 length.
- **Failure trajectories vs successes**: `traj.is_failure` is the binary
  ground-truth label. Failure ones additionally have `traj.failure_segments`
  (`[{start, end, mode}, ...]`) and `traj.load_failure_mask()`.

---

## 3. `BenchmarkTrajectory` (Agilex) — what data is available

```python
traj.task_name              # "candy_in_plate" | "duck_in_bowl" | ...
traj.video_id               # stable string id
traj.num_frames             # int T (clipped to min over all streams)
traj.is_failure             # bool
traj.available_cameras      # ("cam_high", "cam_left_wrist", "cam_right_wrist")
traj.failure_segments       # [{"start": int, "end": int, "mode": str}, ...]
traj.first_gt_failure_frame()  # min(start) or None for success

# Lazy loaders (each opens HDF5 / mmaps cache on call; no caching in the
# trajectory object — the discriminator must cache outputs itself).
traj.load_images(cameras=[...])   # -> {cam: (T, H, W, 3) uint8}
traj.load_states()                # -> (T, D_state) float (sliced per proprio_slice)
traj.load_actions()               # -> (T, D_act) float (sliced per action_slice)
traj.load_failure_mask()          # -> (T,) uint8 (or None for success)
traj.load_failure_segment_index() # -> (T,) int32 (or None for success)
```

**Defaults (Agilex, right-arm 6-DoF setup)**:
`proprio_field="qpos"`, `proprio_slice=slice(7,14)` (right-arm 7-D),
`action_slice=slice(7,13)` (right-arm 6-DoF, no gripper),
`available_cameras` includes `cam_high` (default), often plus wrist cams.

**Tip**: cache encoded features per trajectory using a stable key (e.g.
`(traj.file_path, traj.episode_path or traj.cache_npz_path)`) — the same
success trajectory is encoded once during `fit_on_benchmark` and again
during `score_trajectory`, so caching saves ~50% of encoder time.

---

## 4. Dataset layout

### Raw HDF5 (default)

```
data/agilex/
├── <task>/success_rollout/episode_*.hdf5         # success episodes (root group)
└── failure_annotations/out_by_task/<task>/out.hdf5  # all failure episodes
    └── episodes/<split>/<episode_N>/             # one group per episode
        ├── action                          (T, 14)        float32
        ├── observations/qpos               (T, 14)
        ├── observations/images/<cam>       (T, H, W, 3)   uint8
        ├── annotations/failure_frame_mask  (T,)           uint8
        ├── annotations/failure_segment_index (T,)         int32
        └── attrs.failure_segments_json     '[{start,end,mode}, ...]'
```

`success_rollout/episode_N.hdf5` has the **same fields at the file root**
(no `episodes/<split>/...` indirection), with no annotations.

### Optional `.npy` cache (recommended for repeated runs)

Built once via `bash benchmark/real_world/scripts/build_cache.sh`. Layout:

```
data/.agilex_train_cache/<task>/<sha1>/
├── images_chw.npy       (T, num_cameras, 3, H, W) uint8   center-cropped + resized
├── proprio.npy          (T, D_state) float32
├── actions.npy          (T, D_act)   float32
├── failure_mask.npy     (T,)         uint8
├── failure_segment_index.npy (T,)    int32
└── metadata.json        camera_names, video_id, file_path, failure_segments, ...
```

Cache hits the same `AgilexBenchmarkTrajectory` API (transparent to the
discriminator). Selected automatically by passing `cache_root=...` to
`FailureBenchmark`. **Cache must live under `data/`** (enforced by
`_ensure_local_cache_root`).

The `bash` script is the primary entry — set `IMAGE_SIZE` once during
caching to fix the resolution that *all* downstream discriminators see.

---

## 5. Adding a new discriminator (step-by-step)

### 5.1 Write the discriminator class

```
robosuite/discriminator/<name>/
├── __init__.py
├── <name>_benchmark.py       # MyBenchmarkDiscriminator class
├── <internal modules>.py     # encoder, model, monitor, ...
└── scripts/
    └── run_<name>_real_world_benchmark.sh
```

Skeleton (`<name>_benchmark.py`):

```python
from benchmark.core import BenchmarkTrajectory, DiscriminatorOutput

class MyBenchmarkDiscriminator:
    name = "my_method"

    def __init__(self, *, device="cuda", camera_name="cam_high", **hp):
        self.encoder = ...   # load once
        self.camera_name = camera_name
        self._models_per_task: dict[str, MyModel] = {}
        self._feat_cache: dict[tuple[str, str], np.ndarray] = {}

    def _key(self, traj):  # stable cross-call cache key
        return (str(getattr(traj, "file_path", "")),
                str(getattr(traj, "cache_npz_path", "") or
                    getattr(traj, "episode_path", "")))

    def _encode(self, traj):
        k = self._key(traj)
        if k in self._feat_cache:
            return self._feat_cache[k]
        imgs = traj.load_images(cameras=[self.camera_name])[self.camera_name]
        T = min(int(imgs.shape[0]), int(traj.num_frames))
        feats = self.encoder.encode_images(imgs[:T])  # (T, D)
        self._feat_cache[k] = feats
        return feats

    def fit_on_benchmark(self, trajectories):
        # Per-task split / training / calibration over success trajectories ONLY.
        by_task = {}
        for t in trajectories:
            if t.is_failure: continue
            by_task.setdefault(t.task_name, []).append(t)
        for task, succ in by_task.items():
            # train on success features, calibrate threshold, etc.
            self._models_per_task[task] = train_one_task(succ, self._encode)

    def score_trajectory(self, traj):
        feats = self._encode(traj)
        T = int(feats.shape[0])
        model = self._models_per_task[traj.task_name]
        scores = model.score(feats).astype(np.float32)        # (T,) higher = more anomalous
        preds  = model.predict(scores, T).astype(np.int64)    # (T,) optional binary
        first  = int(np.argmax(preds == 1)) if preds.any() else None
        return DiscriminatorOutput(
            step_scores=scores,
            predictions=preds,
            first_failure_frame=first,
            aux={"task": traj.task_name},
        )

    def close(self):
        self.encoder.close()
```

### 5.2 Write the entry script

`benchmark/real_world/examples/run_<name>.py` (mirror `run_logpZO.py`):

```python
import argparse
from benchmark.core import EvalConfig
from benchmark.real_world import FailureBenchmark
from robosuite.discriminator.<name>.<name>_benchmark import MyBenchmarkDiscriminator

def main():
    args = _parse_args()  # --fail-root, --success-root, --cache-root, --tasks, --save-json, ...
    bench = FailureBenchmark(
        fail_labeled_root=args.fail_root,
        success_root=args.success_root,
        cache_root=args.cache_root,         # or None for raw HDF5
        tasks=args.tasks,
        max_fail_per_task=args.max_fail_per_task,
        max_success_per_task=args.max_success_per_task,
        proprio_field=args.proprio_field,
        proprio_slice=slice(args.proprio_start, args.proprio_stop),
        action_slice=slice(args.action_start, args.action_stop),
    )
    discr = MyBenchmarkDiscriminator(...)
    try:
        discr.fit_on_benchmark(bench.trajectories())
        result = bench.evaluate(discr, EvalConfig(step_binarize_strategy="provided"))
        print(result.summary())
        if args.save_json:
            result.save_json(args.save_json)
    finally:
        discr.close()
```

### 5.3 Bash wrapper

Copy `robosuite/discriminator/logpZO/scripts/run_logpZO_real_world_benchmark.sh`
and rename. Critical knobs to keep configurable via env vars:
`FAIL_ROOT`, `SUCCESS_ROOT`, `CACHE_ROOT`, `TASKS`,
`MAX_{FAIL,SUCCESS}_PER_TASK`, `RUN_NAME`/`OUT_DIR`/`SAVE_JSON`,
`DEVICE`, `CAMERA_NAME`, `SEED`, plus your method-specific hyper-params.

Usage convention:
```bash
RUN_NAME=run_$(date +%Y%m%d_%H%M%S) \
  bash robosuite/discriminator/<name>/scripts/run_<name>_real_world_benchmark.sh
```
Output lands in `checkpoints/<name>/real_world_eval/${RUN_NAME}/benchmark.json`.

---

## 6. `EvalConfig` knobs and what they mean for your method

`EvalConfig` controls **only the metric-computation side**, not the
discriminator. Defaults:

```python
EvalConfig(
    traj_score_aggregator="max",          # max / mean / topk_mean for traj-level score
    traj_score_topk=20,
    event_iou_threshold=0.1,              # event-level matching
    step_binarize_strategy="success_calibrated",
    step_success_percentile=95.0,         # for success_calibrated
    step_fixed_threshold=None,            # for fixed
)
```

### `step_binarize_strategy` — pick deliberately

| Strategy | Source of binary predictions | When to use |
|---|---|---|
| `"provided"` | `output.predictions` from `score_trajectory` | Your method has its **own** threshold (CP, conformal, learned classifier); paper-faithful. |
| `"success_calibrated"` | 95th-percentile of pooled success step-scores (across **all** success trajectories) | Your method only outputs continuous scores, no calibration. Or you want a fair "thresholded at success-95th" comparison. |
| `"fixed"` | `step_fixed_threshold` constant | Ablation. |

Mismatching this with your method silently changes results. **logpZO lesson**:
the sim entry used the default `success_calibrated`, the real-world entry
used `"provided"` — that alone made step-level binary metrics unfair to
compare. Pick one strategy per study and stick to it.

### Trajectory-level aggregation

`traj_score = aggregate(step_scores, mode=...)`. Default `max` is how the
paper-style trajectory AUROC is typically computed. Switch to `topk_mean`
if your scores are jittery and a single-frame max is too noisy.

---

## 7. Output schema (`benchmark.json`)

```jsonc
{
  "discriminator_name": "...",
  "trajectory_level":   { "auroc", "auprc", "best_f1", "tpr_at_fpr_*", ... },
  "trajectory_level_per_task": { "<task>": {... same keys ...} },
  "step_level":         { "frame_auroc", "frame_f1", "frame_precision",
                          "frame_recall", "event_recall", "event_precision",
                          "num_pred_segments", "detection_delay_mean", ... },
  "step_level_per_task": { "<task>": {...} },
  "per_trajectory":     [ { "task_name", "video_id", "is_failure",
                            "trajectory_score", "first_pred_failure_frame",
                            "first_gt_failure_frame", "failure_segments" }, ... ],
  "config":  { ...EvalConfig + source_metadata... },
  "runtime": { "total_seconds", "num_trajectories" }
}
```

Sanity tools (one-liners against the JSON):

```python
# Empirical trajectory-level FPR (= 1 - TNR), critical for any CP method:
fp_rate = sum(1 for r in d["per_trajectory"]
              if (not r["is_failure"]) and r["first_pred_failure_frame"] is not None
             ) / sum(1 for r in d["per_trajectory"] if not r["is_failure"])
```

---

## 8. Lessons / pitfalls (from logpZO + LPB integrations)

**Performance-related**

1. **Frame_AUROC ≠ trajectory_AUROC.** Real-world Agilex failure trajectories
   are visually OOD vs. successes from frame 0 (different recording session,
   lighting, camera pose). Trajectory-level metrics stay strong (~0.98 AUROC),
   but per-frame ranking is harder (~0.7 AUROC). Don't tune for one and report
   the other.
2. **`detection_delay` is signed**: negative means "fired before GT failure
   started". Strongly negative (< -100) almost always means the early-frame
   score is high, not that the model is "predictive". Investigate before
   celebrating.
3. **`num_pred_segments` ≫ `num_gt_segments`** indicates jittery scores. A
   short causal smoothing on `step_scores` (e.g. EMA or 5–10 frame moving
   average) before thresholding usually collapses 10–20× the segment count
   without hurting AUROC.
4. **Conformal threshold sample size matters.** With `calib_fraction=0.2`
   and 100 success trajectories, you get N=20 calibration trajectories.
   That's noisy — quantile estimates can be conservative by 10–20pp. If your
   empirical FPR is way below α, this is why; either bump `calib_fraction`
   or accept that paper's α is a target, not a guarantee.

**Wiring-related**

5. **Always cache encoded features.** Successes are encoded twice (fit + score)
   if you don't. Cache key must be cross-storage-stable: prefer
   `(file_path, cache_npz_path or episode_path)` over object identity.
6. **`load_images` is expensive.** It returns *all frames* of *all requested
   cameras* as a numpy array. Always pass `cameras=[self.camera_name]`, not
   the default (which loads every available camera).
7. **Trim to `traj.num_frames`** after loading any stream. `num_frames` is
   already the min over all aligned modalities; HDF5 raw lengths can differ.
8. **Don't trust `traj.failure_segments` for success trajectories** — it's
   `[]`. Only check `traj.is_failure` first, then load mask/segments.
9. **`fit_on_benchmark` sees the whole pool.** Filter `traj.is_failure` out
   yourself; the framework does not pre-split. logpZO/LPB style: trajectory-
   level seeded train/calib split for conformal validity.

**Reproducibility**

10. **Fix seed at every random surface**: `numpy.random.default_rng(seed)`
    for splits, `torch.manual_seed(seed)` for flow training, `seed` argument
    in encoder data loaders. The `--seed` CLI flag is convention.
11. **Save full calibration metadata in the JSON** (alpha, num_calib, threshold
    distribution). Future-you will need it for plots and ablations. `aux`
    in `DiscriminatorOutput` is *not* persisted — write hyperparams via
    `EvalConfig` + a side-car JSON or the bash script header.

**Comparison fairness**

12. **Same `EvalConfig` across baselines.** In particular, `step_binarize_strategy`
    must match. If two methods provide their own thresholds, use `"provided"`
    for both; if one doesn't, switch both to `"success_calibrated"`.
13. **Same `--max-{fail,success}-per-task`** across runs you intend to
    compare. Calibration-set size affects conformal thresholds non-trivially.
14. **Same `CACHE_ROOT`** (or no cache for both). Different cache image_size
    silently changes the encoder input.

---

## 9. Quick reference: file paths

| What | Where |
|---|---|
| Core protocol | `benchmark/core/discriminator.py` |
| Trajectory base | `benchmark/core/trajectory.py` |
| Metrics | `benchmark/core/metrics.py` |
| Real-world skin | `benchmark/real_world/__init__.py`, `trajectory.py`, `loader.py` |
| Cache builder | `benchmark/real_world/examples/build_cache.py` (+ `scripts/build_cache.sh`) |
| Dummy smoke test | `benchmark/real_world/examples/run_dummy.py` |
| Reference impl | `robosuite/discriminator/logpZO/{logpZO_benchmark.py, monitor.py, scripts/}` |
| Default data root | `data/agilex/` (raw), `data/.agilex_train_cache/` (cache) |
| Default output root | `checkpoints/<method>/real_world_eval/<run_name>/benchmark.json` |
