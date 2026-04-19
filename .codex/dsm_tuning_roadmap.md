# DSM Tuning Roadmap

Working doc for the DSM detector effort. Pairs withc`discriminator_progress.md` (background & what's already been tried) andc`dsm_phase1_refactor_prompt.md` (the active coding-agent task).

## Context

`lpb_score` is currently broken: `chunk_margin ≈ 1e-8`, success and fail λcdistributions overlap, λ even reverses direction during true failure on `PickPlaceCan`. Root cause is **implementation drift from the DSM theory**, not a conceptual dead end:

- no noise at inference → DSM collapses to an identity autoencoder.
- `normalize_latent` is identity; μ/σ buffers are never applied.
- single-σ training; encoder frozen; no gradient pressure on the `c` condition embedding.
- left padding creates cold-start false positives.

Evaluation is now solid: `data/utils/benchmark/` provides trajectory-level AUROC/AUPRC and step-level frame AUROC / event recall / detection delay.

## Execution rule

All phases below will be tried. There is **no performance gate between phases** — we are not skipping or stopping based on benchmark numbers. Each phase is run, measured on the benchmark, and the result is recorded; the user reviews the full table afterwards. Max one iteration of hyperparameter tuning per phase.

## Phase 1 — DSM correctness + conditional gradient pressure  [ACTIVE]

Goal: make `lpb_score` a real DSM, not an identity autoencoder, and make the class condition `c` carry real gradient signal. Single refactor, all always-on.

- Real `normalize_latent` applied (positive-only μ/σ, computed in a warmup pass).
- Inference adds noise and uses Monte-Carlo averaging; score is point-wise Fisher divergence via Tweedie estimator.
- Multi-σ training (NCSN log-uniform) with σ fed into the network through a sinusoidal embedding in the condition MLP.
- Cold-start mask over the first `W-1` frames of every trajectory.
- Auxiliary conditional contrastive loss (margin hinge on `d0 − d1` per class), weight 0.1, always on.

The Phase 1 refactor also pre-wires Phase 2 (LoRA) and Phase 3 (EWMA) scaffolding — defaults keep them off, env vars switch them on. After this refactor, Phase 2 and Phase 3 are bash flag flips, no further code changes.

Measure and record: `chunk_margin_mean/std`, trajectory AUROC, frame AUROC, event recall @ IoU 0.3, `detection_delay_median`.

Full instructions: `dsm_phase1_refactor_prompt.md`.

## Phase 2 — LoRA adaptation  [flag flip]

Already wired in Phase 1. Trigger with env vars:

```bash
USE_LORA=1 TRAIN_ENCODER=1 LORA_RANK=16 LORA_ALPHA=32.0 \
  bash .../scripts/train_lpb_score_dsm.sh
```

- LoRA adapts the condition-path layers of the encoder. Flow head stays frozen.
- Retrain the Phase 1 objective (DSM + aux contrastive) with LoRA on, one seed.

Measure the same benchmark metrics and record.

## Phase 3 — Time-aware aggregation  [flag flip]

Already wired in Phase 1. Trigger at benchmark time — no retrain needed:

```bash
LAMBDA_MODE=ewma EWMA_ALPHA=0.9 \
  bash .../scripts/run_lpb_score_benchmark.sh
```

Current `mean` / `max` aggregators are monotone in time and cannot reflect robot recovery. EWMA `λ_t = α · λ_{t-1} + (1 - α) · u_t` allows λ to decrease.

- Sweep α in {0.80, 0.90, 0.95}.
- Re-evaluate on fail trajectories that contain a recovery segment.

Record event recall @ IoU 0.3 and `detection_delay_median` for each α.

## Ablation matrix

Phase 1 ships one configuration with everything always-on (normalize + MC-noise + multi-σ + cold-start mask + aux contrastive). Phase 2/3 are flag flips on top. All rows are run regardless of earlier results.

| Run | Phase 1 defaults | LoRA | EWMA α |
|---|---|---|---|
| P1   | ✓ | — | — |
| P2a  | ✓ | rank 8  | — |
| P2b  | ✓ | rank 16 | — |
| P3a  | best of P1/P2 | same | 0.80 |
| P3b  | best of P1/P2 | same | 0.90 |
| P3c  | best of P1/P2 | same | 0.95 |

Debug-only ablations (to verify Phase 1 is well implemented, not for headline numbers): flip `AUX_WEIGHT=0`, `normalize_enabled=false`, or single-σ training off via config overrides.

## Out of scope for this roadmap

- Contrastive energy / NCE (Direction 2).
- nnPU / KLIEP / uLSIF density-ratio estimation (Direction 3).
- Conformal anomaly detection.
- Expanding success data to tasks beyond `PickPlaceCan`.
- New benchmark metrics or tasks.
