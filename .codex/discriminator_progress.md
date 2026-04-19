# Failure Detector: Progress & Roadmap

Short handoff doc for the next conversation. Read AGENTS.md first.

## 1. Goal
Offline failure detector / discriminator over multi-task robosuite rollouts.
Output per-step anomaly scores + temporally aggregated score + binary decisions
+ visualization. See `prompt/discriminator.md` for the full problem statement.

## 2. Environment
- Code lives on remote **192.168.15.76**, user `dodo`, password `123456`
  (see `ssh/config.json`). All paths below are on the remote.
- Repo root: `/home/dodo/Documents/DAggar/robosuite`
- Conda env: `daggar` (`/home/dodo/miniconda3/envs/daggar/bin/python`)
- Logging: WandB offline, `WANDB_NAME=songgao-personal`.

## 3. Approaches Tried (what works, what doesn't)

| Approach | Location | Result |
|---|---|---|
| **FLOAT OT** (arxiv 2510.02298) | — | Good aggregate trajectory score; cannot detect recovery (monotonic); weak theory. |
| **Naive dynamics-model composite**: fk + te + pc + dy | — | Works reasonably; no theory backing. |
| **BCE/DICE PU head** | `dice` branch, `robosuite/discriminator/lpb_dice` | Elegant (occupancy-ratio → reward) but empirically poor: BCE not sensitive at failure onset; fail_rollout contains many good transitions → heavy label noise. |
| **Denoising Score Matching (current)** | `robosuite/discriminator/lpb_score` | **Broken** — see §5. |

## 4. Current code — `lpb_score`
- `core/model.py` — `ChunkConditionedDSM` + `DSMModel` wrapper, AdaLN temporal Conv1d.
- `core/dsm_discriminator.py` — inference-time scorer, T3 combo `α·z(chunk_energy) − β·z(chunk_margin)`.
- `core/trainer.py` — plain chunk reconstruction MSE with `add_noise=True`.
- `config/train.yaml` — current training config: `window_size=8`, `noise_scale=0.08`, `trainable_encoder=false`, **`lora.enabled=false`**.
- Latest result dir: `checkpoints/multitask_6/lpb_dipole-new-v4/visualize/run_20260419_155012/`.

## 5. Root causes of lpb_score's poor performance (confirmed)

Evidence: `summary.json` has `chunk_positive_mean ≈ 5.5e-6`, `chunk_margin_mean ≈ 1.5e-8`, `chunk_margin_std` pinned at `1e-6` floor. Threshold distribution plots show success and fail λ histograms nearly identical. In `traj00.pdf`, λ goes **negative** during actual failure phase (direction reversed).

1. **Inference uses `add_noise=False`** → DSM becomes identity: `g(x, c) ≈ x` for both `c=0` and `c=1`. Reconstruction energy ≈ 0, margin ≈ 0. The Fisher-like interpretation `||x̂⁰−x̂¹||² ≈ σ⁴·||∇log p(c=0) − ∇log p(c=1)||²` requires noise. See `DSMModel.compute_fisher_score_from_latent_window`.
2. **`normalize_latent` is identity** — mean/var are computed, stored, never applied. README §5 is aspirational.
3. **LoRA disabled** — config has `lora.enabled: false`, so encoder is fully frozen policy-encoder; its feature space is tuned for action prediction, not class separation.
4. **Objective doesn't force `c` to matter** — training minimizes MSE; conditional embedding has no gradient pressure to differentiate classes when the network can just copy its input.
5. **Left-padding** (`_build_latent_windows` pads with `latents[0]`) creates cold-start false positives.
6. **No step-level GT historically** — evaluation was trajectory-level only. (Now fixed, see §7.)

## 6. Roadmap (ordered by ROI)

Direction 1 (minimal, first): **Use DSM correctly**
- Enable `normalize_latent` with positive-only μ/σ.
- At inference add noise and Monte-Carlo average: `x̃ = x + σε`, compute `s_c(x̃) ≈ (g(x̃,c) − x̃)/σ²`, score = `‖s_0 − s_1‖²` (pointwise Fisher).
- Train multi-σ (NCSN-style, `σ ~ LogUniform`).
If `chunk_margin` stays at 1e-8 after this, §Direction 2 is required.

Direction 2: **Drop denoising, learn a contrastive energy**
- `E_φ(x_t^window, k)` trained with NCE; negatives are **time-shuffled / cross-trajectory windows**, not `fail_rollout` (which is label-noisy).
- Immune to "early-success ≈ late-failure" visual overlap.

Direction 3: **nnPU / density-ratio on PU data**
- `nnPU` with `max(0, R_U^− − π R_P^−)` correction.
- Or direct DRE (KLIEP / uLSIF) to estimate `p_fail / p_succ`.

Direction 4: **Time-aware scoring for recovery detection**
- EWMA with decay; CUSUM/GLRT; conformal p-values.
- So λ can *decrease* when the robot recovers.

Direction 5: **Feature space**
- Enable the existing LoRA (rank 8–16) ONLY after switching objective (else LoRA collapses to the trivial solution again).
- Or a small projection head `h = MLP(z_t)` trained only on detector loss.

Direction 6 (prerequisite, DONE): **Evaluation**
- Step-level GT labels now exist: `data/utils/fail_rollout/<task>/out.hdf5`
  (see `data/utils/data_README.md`).
- Benchmark built — see §7.

## 7. Benchmark (just built)

Location: `data/utils/benchmark/` on remote. See `data/utils/benchmark/README.md`.

```python
from data.utils.benchmark import FailureBenchmark, BenchmarkTrajectory, DiscriminatorOutput

class MyDisc:
    name = "my_method"
    def score_trajectory(self, traj: BenchmarkTrajectory) -> DiscriminatorOutput:
        imgs   = traj.load_images(cameras=["agentview"])  # dict cam -> (T,H,W,3) uint8
        states = traj.load_states()                       # (T, 71) float64
        actions= traj.load_actions()                      # (T, 7)  float32
        scores = my_fn(...)                               # (T,) higher = failure-like
        return DiscriminatorOutput(step_scores=scores)

bench = FailureBenchmark(
    fail_labeled_root="/home/dodo/Documents/DAggar/robosuite/data/utils/fail_rollout",
    success_root="/home/dodo/Documents/DAggar/robosuite/data/utils/success_rollout",
    tasks=["PickPlaceCan"],
)
result = bench.evaluate(MyDisc())
print(result.summary())
result.save_json("/tmp/bench.json")
```

- Discovered trajectories: 11 labeled-fail + 10 success on PickPlaceCan; 20 fail each on Bread/Cereal/Milk (no success data yet).
- Metrics:
  - Traj-level: AUROC, AUPRC, best_f1, TPR@FPR, FPR@TPR (default aggregator = `max` of step_scores).
  - Step-level (on fail traj only): frame_auroc, frame_auprc, event_recall@IoU, detection_delay (mean/median/p25/p75), frame_f1/iou.
- Binarization for step metrics: default `success_calibrated` at 95th percentile of pooled success scores. Alternatives: `provided` (discriminator supplies) / `fixed` (user threshold).
- Smoke test passed with `RandomDiscriminator` — AUROC ≈ 0.49 as expected.
- Entrypoint: `bash data/utils/benchmark/scripts/run_benchmark_example.bash`.

Data schema details (already encoded in the benchmark, but useful for new discriminators):
- `fail_rollout/<task>/out.hdf5 → /demos/video_<id>/{actions, states, observations/<cam>/images, annotations/failure_frame_mask, annotations/failure_segment_index}` + `failure_segments_json` attr.
- `success_rollout/<task>/video_manifest.jsonl` → `hdf5_path + demo_key` into `{actions, states, observations/<cam>/images}`.
- No proprio field — only `states` shape `(T, 71)`.

## 8. Concrete next actions (in order)

1. **Wrap `lpb_score` as a benchmark Discriminator** and measure current baseline. File skeleton: `data/utils/benchmark/examples/run_lpb_score.py`. Required to quantify the current floor before changes.
2. Implement **Direction 1** (correct DSM inference): a 1-day patch to `compute_fisher_score_from_latent_window` + enable `normalize_latent` + multi-σ training. Rerun benchmark.
3. If margin still collapses → implement **Direction 2** (contrastive energy) from scratch in a new module, likely `robosuite/discriminator/lpb_ce/`.
4. Add **Direction 4** (EWMA / CUSUM) on top of whichever step_score survives, to handle recovery.
5. Expand success data to other tasks (Lift / PickPlaceBread/Cereal/Milk / Stack) so trajectory-level AUROC is meaningful on all tasks.

## 9. Key files to cite in the next conversation

- `prompt/discriminator.md` — problem setup.
- `prompt/discriminator_progress.md` — **this file**.
- `AGENTS.md` — collaboration rules.
- On remote: `robosuite/discriminator/lpb_score/{core/model.py, core/dsm_discriminator.py, config/train.yaml}` — current (broken) impl.
- On remote: `data/utils/benchmark/README.md` — benchmark API.
- On remote: `data/utils/data_README.md` — labeled data schema.
