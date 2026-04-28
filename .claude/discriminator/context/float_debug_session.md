# FLOAT Debug Session

Date: 2026-04-27
Branch: `discriminator-baseline`
Working dir: `/home/dodo/Documents/DAggar/robosuite`
Paper: ARMADA / FLOAT, https://arxiv.org/pdf/2510.02298 (Yu et al., 2025)
Reference impl: `tmp/armada/float/` (`float.py`, `util.py`)
Our impl: `robosuite/discriminator/float/`

## Symptom

Old benchmark output `checkpoints/float/eval/run_20260421_214005/benchmark.json`:

- Trajectory-level **AUROC = 0.62** (logpZO baseline on the same data: 0.96).
- Frame-level AUROC = 0.54 (≈ random).
- `first_pred_failure_frame = 1` for almost every trajectory (success and failure
  alike) → the calibrated Λ is crossed at the very first non-trivial prefix.
- Per-task: PickPlaceCan 0.54 / Bread 0.62 / Cereal 0.62 / Milk 0.74.

## What I checked

### 1. Algorithm vs. paper Eq.(5)/(6) — implementation is correct

`robosuite/discriminator/float/float_core.py`:

- `cosine_cost_matrix(x, y, use_similarity_cost=False)` returns `1 - cos(x,y)` —
  matches `cosine_distance` in `tmp/armada/float/util.py:20`.
- `sinkhorn(...)` is a textbook log-domain Sinkhorn; equivalent to
  `ot.sinkhorn(a, b, c, reg)` used in armada (we just don't pull in POT).
- `FLOATComputer._lambda_single_expert`: uniform marginals `a = 1/l_n`,
  `b = 1/t_0`, `λ = sum(p * c)`. Matches Eq.(5).
- `FLOATComputer.lambda_index = min_n λ_n`. Matches Eq.(6).

`float_benchmark.py`:

- Per-task expert bank built from success trajectories only.
- Threshold Λ via `(1 − δ)`-percentile over LOO success λ at full length —
  matches paper's "calibrate on all successful rollouts".
- LOO at scoring time (`exclude_idx`) avoids self-match.
- `step_stride=8`: λ recomputed every 8 frames, held in between (compute
  optimization, not algorithmic).

**Conclusion**: the FLOAT math/code is faithful to the paper. No algorithmic bug.

### 2. Deviations from the paper that don't matter for this benchmark

- **Expert padding to `l_max`** (Appendix A): paper pads all experts to
  `l_max := max_n l_n` by repeating the last frame. In our PickPlaceX data
  every trajectory is exactly 400 frames, so padding is a no-op.
- **Ta = 8 chunking**: paper uses policy embeddings every Ta=8 steps.
  Implementation-side we just stride at t-level which is equivalent for
  scoring purposes.

### 3. Real difference vs. paper — encoder

Paper §IV-C / Table II uses **DINOv2 ViT-B/14 + a trained linear head
`[768, 192, 64]`** as the policy embedding. The head is jointly trained with
the diffusion policy, so the embedding is task-discriminative.

Our impl uses the **raw DINOv2 CLS embedding (768-d)**, no head. The original
armada code (`float.py:535`) calls `policy.extract_latent(obs_dict)` —
i.e. the *policy's* internal latent, not raw DINOv2.

So this is the most consequential intentional simplification in the port,
and is already documented in `float/README.md`.

## Diagnostic experiments

Scripts: `/tmp/float_diag.py`, `/tmp/float_diag2.py`, `/tmp/float_diag3.py`
(8–12 trajectories per condition for PickPlaceCan, LOO).

### Sanity: cosine cost ranges look healthy

```
within-success cost matrix (success vs success, agentview):
  shape=(400,400) min=0.01 max=0.33 mean=0.12
```

Not collapsed; embeddings have non-trivial spread.

### Lambda separation between SUCC/FAIL — really weak with raw DINOv2

Pairwise AUC = `P(λ_fail > λ_succ)` over all (succ, fail) pairs.
> 0.5 means failure looks more anomalous (correct direction).

| Camera | Full-traj λ AUC | Max-prefix λ AUC |
|---|---|---|
| agentview            | 0.68 | **0.29 (inverted!)** |
| robot0_eye_in_hand   | 0.58 | **0.69** |
| agent + wrist concat | 0.56 | **0.71** |
| agent + wrist + front concat | 0.54 | 0.68 |

Key observations:

1. With **agentview only + max-over-prefix aggregation** (the benchmark's
   default `traj_score_aggregator="max"`), the score actually *anti-correlates*
   with failure — which explains the 0.62 AUROC ceiling. Successful rollouts
   have more natural variation between each other, so the max prefix-OT cost
   for a success often exceeds that of a failure (which "blends in").
2. **Wrist camera** (`robot0_eye_in_hand`) flips this, giving a real signal
   (max-prefix AUC = 0.69).
3. **Concatenating agentview + wrist** is best in this small experiment
   (max-prefix AUC = 0.71), wrist alone is close. Three cameras adds noise.
4. Full-traj λ alone is weak under any camera; the max-over-time aggregation
   at the benchmark layer is doing real work.

### λ(t0) trajectories don't show clean monotone decay

For 8 success and 4 fail PickPlaceCan trajectories at t0 ∈
{2, 4, 8, 16, 32, 64, 128, 200, 300, 400} the values stay in [0.07, 0.10] with
no clean separation between classes at any specific t0. There's no
"early prefix explosion" — the calibrated Λ ≈ 0.08 just sits roughly inside
the range of step-level scores for both classes.

## Side issue (NOT a FLOAT bug, but spotted while debugging)

`data/utils/success_rollout/<task>/video_manifest.jsonl` files reference
`hdf5_path` under `/home/dodo/data1/robosuite/raw/...`, but actual files are
at `/home/dodo/data1/gy/robosuite/raw/...`. PandaPickPlaceCan confirmed; the
other 3 tasks (Bread, Cereal, Milk) have the same `/data1/robosuite/raw/`
prefix and likely need the same `/data1/gy/robosuite/raw/` rewrite.

The 2026-04-21 benchmark run still found 200 trajectories — data must have
moved between then and now. Independent of the FLOAT debugging task.

## Decision

User's stance: implementation is correct, the encoder gap is acceptable as a
documented limitation of the port. **No code changes from this session.**

If we ever want to push FLOAT past the encoder ceiling, the path is:

1. Plug a discriminative encoder via `EmbeddingEncoder` Protocol —
   either a trained policy latent (matches paper) or a manipulation-pretrained
   ViT (R3M / VC-1 / theia / MVP).
2. Or extend `FloatBenchmarkDiscriminator` to accept multiple cameras and
   concat embeddings (wrist gives most of the signal in our diagnostics).

## File pointers

- Algorithm: `robosuite/discriminator/float/float_core.py`
- Adapter:   `robosuite/discriminator/float/float_benchmark.py`
- Encoder:   `robosuite/discriminator/float/float_dino_encoder.py`
- Runner:    `data/utils/benchmark/examples/run_float.py`
- Script:    `robosuite/discriminator/float/scripts/run_float_benchmark.sh`
- Reference: `tmp/armada/float/{float.py,util.py}`
- Old run:   `checkpoints/float/eval/run_20260421_214005/benchmark.json`
