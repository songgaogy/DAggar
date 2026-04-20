# logpZO — FAIL-Detect Failure Monitor

Implements the **logpZO** variant of FAIL-Detect (arXiv:2503.08558) for
runtime OOD detection in imitation-learning policies.

## Algorithm in one screen

1. **Stage 1 — Density estimation.** A RealNVP normalizing flow `f: z -> s`
   is trained on DINOv2 image embeddings of **success** trajectories.
   Training loss: `-mean log p(s)` under the standard change-of-variables
   formula.
2. **Stage 2 — Conformal threshold.** Given a tolerance `alpha in (0, 1)`
   we compute
   `tau = Quantile(S, ceil((N+1)(1 - alpha))/N)`
   over step-level scores `S = { -log p_Z(f^{-1}(s)) }` on the success
   pool. A frame is flagged when its score exceeds `tau`.

The score is the logpZO term only (base-Gaussian log-prob of the inverted
latent), dropping the Jacobian contribution — this is the paper-recommended
variant.

## Files

| File | Role |
|------|------|
| `flow_model.py` | RealNVP flow with `log_prob` (full) and `log_pz_only` (logpZO). |
| `monitor.py` | `FAILDetectMonitor` — `fit_score_model`, `calibrate_conformal_threshold`, `score_features`. |
| `logpZO_benchmark.py` | Adapter to the shared `data/utils/benchmark` harness (per-task flow + threshold). |
| `scripts/run_logpZO_benchmark.sh` | Bash entry point with safe defaults. |

## How to run

```bash
# From repo root on the remote server.
bash robosuite/discriminator/logpZO/scripts/run_logpZO_benchmark.sh
```

Common overrides:

```bash
TASKS="PickPlaceCan" \
MAX_SUCCESS_PER_TASK=20 \
MAX_FAIL_PER_TASK=10 \
EPOCHS=30 \
ALPHA=0.1 \
  bash robosuite/discriminator/logpZO/scripts/run_logpZO_benchmark.sh
```

Output `benchmark.json` is written under
`checkpoints/logpZO/eval/<run_name>/benchmark.json` and conforms to
`data/utils/benchmark/protocol.md`.

## Visualization

Render random failure trajectories into VS-Code-friendly mp4s (H.264 /
yuv420p, red border on flagged frames) plus a multi-page PDF showing the
score curve vs the CP threshold.

```bash
# Default: PickPlaceBread, 4 sampled fail trajectories.
bash robosuite/discriminator/logpZO/scripts/visualize_logpZO.sh

# Single-task overrides:
TASK=PickPlaceCan NUM_TRAJS=6 SEED=42 \
MAX_SUCCESS_PER_TASK=30 MAX_FAIL_PER_TASK=20 EPOCHS=30 \
  bash robosuite/discriminator/logpZO/scripts/visualize_logpZO.sh
```

Outputs:
* `checkpoints/logpZO/viz/<run_name>/videos/<video_id>.mp4`
* `checkpoints/logpZO/viz/<run_name>/logpZO_scores.pdf`
