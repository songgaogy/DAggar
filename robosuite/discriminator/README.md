## FLOAT in `robosuite/discriminator`

This module implements FLOAT (Failure detection based on Optimal Transport) for PandaLift data and aligns the runtime logic with the official ARMADA failure detector as closely as possible within this codebase.

## What This Implementation Provides

- A reusable OT core (`sinkhorn`, cosine cost, trajectory utilities).
- An ARMADA-style FLOAT evaluator with:
  - expert candidate retrieval,
  - online expert rematching,
  - cumulative greedy OT-cost thresholding.
- Policy-latent extraction from your Flow Matching policy checkpoint (`./robosuite/policy`, `./checkpoints/PandaLift/flow/BC_warmup`).
- Training / evaluation scripts and Hydra configs.

## Algorithm Summary

Given:

- expert trajectories `T_e,n`
- rollout prefix `T_b[1:t]`
- policy embedding function `phi(o)`

FLOAT computes:

1. **Embeddings**
- `F_e,n = { phi(o_e,i) }`
- `F_b,t = { phi(o_b,j) }` for prefix

2. **OT matching cost**
- cosine-based cost matrix `C` between expert and rollout embeddings
- entropic Sinkhorn plan `P*`
- transport cost `lambda_n = sum_ij P*_ij * C_ij`

3. **FLOAT index**
- implementation supports min-over-experts behavior and ARMADA-style matched-expert behavior
- the main eval path uses ARMADA-style matched expert and cumulative greedy OT-cost over time

4. **Threshold calibration**
- collect episode-level success costs
- threshold `Lambda = percentile(1 - delta)` with `delta` in percent (default 10)

5. **Detection**
- failure if cumulative OT cost exceeds `Lambda`

## Policy Latent Source

Policy embeddings are extracted from Flow policy internals, not raw states:

- checkpoint loaded from `policy.ckpt`
- latent source: conditional token from Flow backbone (`flow_cond_token`)
- code: `robosuite/discriminator/float_policy_latent.py`

Pipeline per trajectory:

1. Load `states` and camera images from HDF5.
2. Build proprio from states using `PandaLiftProprioExtractor`.
3. Build history windows (`history_len`) and sample every `Ta`.
4. Run Flow backbone and extract conditional token embedding.

## Code Structure

- `float_core.py`
  - generic OT and base FLOAT utilities
  - `Trajectory`, `IdentityEncoder`, `TorchEncoderWrapper`, `sinkhorn`
- `float_official.py`
  - ARMADA-style offline evaluator
  - candidate matching, rematching, cumulative greedy OT-cost
- `float_policy_latent.py`
  - Flow checkpoint loading and policy latent extraction
- `float_data.py`
  - HDF5 loading utilities
  - `PolicyTrajectory` (`states + images + actions`)
- `eval_discriminator.py`
  - validation/evaluation entrypoint
- `train_discriminator_intervention.py`
  - threshold calibration + split metrics + artifact saving
- `config/*.yaml`
  - Hydra configs

## Expected Data Format

Each demo must provide:

- `demos/<demo_key>/states`
- `demos/<demo_key>/actions`
- `demos/<demo_key>/observations/<camera_name>/images`

Default camera is `agentview`.

## Quick Start

### 1) Evaluate

```bash
conda activate daggar
bash robosuite/run/eval_float.sh
```

This writes:

- `outputs/float_eval/summary_quick.json`

### 2) Train/Calibrate

```bash
conda activate daggar
bash robosuite/run/train_discriminator_intervention.sh
```

This writes:

- `float_summary_<timestamp>.json`
- `float_calibration_<timestamp>.npz`

## Main Config Knobs

`robosuite/discriminator/config/eval_discriminator.yaml`

- `float.ta`: detector cadence (official-style step aggregation)
- `float.to`: temporal window size used by FLOAT matching logic
- `float.num_expert_candidates`: number of candidate experts before rematching
- `float.sinkhorn_reg`, `float.max_iter`, `float.tol`: OT solver settings
- `policy.ckpt`: Flow checkpoint path
- `policy.history_len`: window length used for latent extraction (`-1` means infer from checkpoint)
- `labels.fail_tail_ratio`: fail labeling rule for offline validation (default 0.2)

## Output Metrics

`summary_*.json` includes:

- split counts
- threshold and delta
- fail validation metrics (`accuracy`, `precision`, `recall`, `f1`, `fp`, `fn`, ...)
- success false alarm stats
- lambda separation stats
- runtime metadata (`Ta`, `To`, `policy_ckpt`, `policy_latent_source`)

## Unit Tests

```bash
conda activate daggar
python -m pytest -q tests/test_float.py
```

Covers:

- Sinkhorn marginal constraints
- matching vs mismatching lambda behavior
- percentile threshold behavior
- rewind timestep behavior

## Remaining Differences vs Official ARMADA

This implementation is intentionally close, but not byte-identical to `robosuite/armada/failure_detector`:

- It runs offline on stored HDF5 trajectories instead of live async environment hooks.
- It uses the Flow policy latent (`flow_cond_token`) from `robosuite/policy`, not ARMADA diffusion policy internals.
- Current dataset setup is single-camera (`agentview`) unless multi-camera data is provided.
- If rollout-success data is missing, calibration falls back to expert-surrogate success.

These are explicit engineering choices for compatibility with this repository and your data pipeline.
