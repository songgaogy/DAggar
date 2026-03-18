## FLOAT-DINO in `robosuite/discriminator`

This module implements a FLOAT baseline with pretrained DINOv2 image embeddings.

## What This Implementation Provides

- A reusable OT core (`sinkhorn`, cosine cost, trajectory utilities).
- An ARMADA-style FLOAT evaluator with:
  - expert candidate retrieval,
  - online expert rematching,
  - cumulative greedy OT-cost thresholding.
- Image-only embedding extraction from pretrained DINOv2.
- Training / evaluation scripts and Hydra configs.

## Algorithm Summary

Given:

- expert trajectories `T_e,n`
- rollout prefix `T_b[1:t]`
- image embedding function `phi(o_t) = DINOv2(image_t)`

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

## DINOv2 Embedding Source

Embeddings are extracted from a pretrained DINOv2 ViT backbone:

- code: `robosuite/discriminator/float_dino/dino_v2_latent.py`
- latent source: normalized CLS token
- modality: image only
- no action input
- no proprio / state fusion

Pipeline per trajectory:

1. Load `states`, `actions`, and one camera stream from HDF5.
2. Ignore action modality during encoding.
3. Center-crop and resize each frame to `policy.image_size`.
4. Run DINOv2 backbone and extract CLS token.

## Code Structure

- `float_core.py`
  - generic OT and base FLOAT utilities
  - `Trajectory`, `IdentityEncoder`, `TorchEncoderWrapper`, `sinkhorn`
- `float_official.py`
  - ARMADA-style offline evaluator
  - candidate matching, rematching, cumulative greedy OT-cost
- `dino_v2_latent.py`
  - pretrained DINOv2 loading and image-only embedding extraction
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
bash robosuite/run/eval_float_dino.sh
```

This writes:

- `outputs/float_dino_eval_pickplacecan/summary.json`

### 2) Train/Calibrate

```bash
conda activate daggar
bash robosuite/run/train_float_dino.sh
```

This writes:

- `float_dino_summary_<timestamp>.json`
- `float_dino_calibration_<timestamp>.npz`

## Main Config Knobs

`robosuite/discriminator/config/eval_discriminator.yaml`

- `float.num_expert_candidates`: number of candidate experts before rematching
- `float.sinkhorn_reg`, `float.max_iter`, `float.tol`: OT solver settings
- `policy.pretrained_path`: DINOv2 pretrained weight path
- `policy.model_name`: `vit_small` / `vit_base` / `vit_large`
- `policy.image_size`: DINOv2 input resolution, default `518`
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
