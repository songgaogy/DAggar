# D3-Disc — Dichotomous Density-ratio Discriminator

Sibling package to `robosuite/discriminator/lpb`. Implements an AIRL-flavoured
extension of the LPB KNN failure detector:

```
lambda_t = (1 + w) * d^2(phi_t, D+) - w * d^2_weighted(phi_t, D-)
```

where `phi_t` is the 256-D `MultiModalFlowPolicy.encode_context` output,
`D+` is a pooled success KNN bank, and `D-` is a **F3-cleaned** fail KNN bank
whose bank-points are softly re-weighted by their distance to `D+` so that
expert-like frames in fail trajectories are rejected:

```
w_j^- = sigmoid(beta * (d^2_knn(phi_j, D+) - kappa))
```

With `omega = 0` the detector exactly degenerates to LPB's 1-NN baseline
(same code path, neg branch skipped).

## Design docs

* `.claude/discriminator/conversation_context.md` — background + locked decisions D1–D12.
* `.claude/discriminator/sfv_discriminator_design.md` — full math spec.
* `.claude/discriminator/implementation_plan.md` — step-by-step.

## Layout

```
d3disc/
  encoder.py           # FlowMultiEncoderWrapper — reuses data/.lpb_score_cache
  filter.py            # knn_sqdist, compute_f3_weights, weighted_knn_sqdist
  detector.py          # D3Detector: fit_banks -> calibrate -> score
  dataset.py           # LatentFlowDynamicsDataset — transitions from cached z_t + HDF5 (s,a)
  trainer.py           # D3DynamicsTrainer — trains the predictor on cached latents
  train_dynamics.py    # CLI entry: trains DynamicsPredictor (encoder stays frozen)
  dynamics_feature.py  # D3FeatureExtractor — inference-time 3-concat feature
  d3_benchmark.py      # D3BenchmarkDiscriminator (FailureBenchmark adapter)
  visualize.py         # D3Visualizer — mp4 + pdf
  scripts/
    train_d3_dynamics.sh     # train the latent-dynamics predictor
    run_d3_benchmark.sh      # fit + evaluate end-to-end
    visualize_d3.sh          # per-task diagnostic
    ablate_omega_beta_k.sh   # sweep omega x k x beta
```

## Two feature modes

- **Raw z_t** (default when `DYN_CKPT` unset). 256-D flow_multi
  `task_scene_cond` per frame. Fast, no training needed. Fails to fold
  action/dynamics info into the feature.
- **LPB-style projected feature** (when `DYN_CKPT` points at a trained
  `d3_dynamics.pt`). Runs the cached `z_t` through the trained
  `DynamicsPredictor`'s projection layers:
  `phi_t = [obs_proj(z_t), proprio_proj(s_t), mean(action_proj(a_{t:t+h}))]`.
  1536-D by default (`3 x d_model`). Dynamics loss regularizes the feature
  geometry and the action-chunk projection makes the feature action-aware.
  The `DynamicsPredictor` architecture is reused verbatim from
  `lpb.model.DynamicsPredictor`; only the input `latent_dim` switches from
  512 (ResNet-18) to 256 (flow_multi aggregator).

## Run

```bash
# ---- 1) Train the latent-dynamics predictor (optional, once). ----
# Uses cached flow_multi latents; encoder stays frozen.
# SUCC_NUM/EXPERT_NUM cap how many demos per kind go into training.
WANDB_MODE=offline WANDB_NAME=songgao-personal \
    EPOCHS=30 BATCH_SIZE=256 SUCC_NUM=0 EXPERT_NUM=0 \
    RUN_NAME="d3dyn_v1" \
    bash robosuite/discriminator/d3disc/scripts/train_d3_dynamics.sh
# -> checkpoint: checkpoints/d3disc/dynamics/d3dyn_v1/d3_dynamics.pt

# ---- 2) Full benchmark (fit + evaluate). ----
# With dynamics ckpt: LPB-style 1536-D projected feature.
DYN_CKPT="checkpoints/d3disc/dynamics/d3dyn_v1/d3_dynamics.pt" \
WANDB_MODE=offline WANDB_NAME=songgao-personal \
    OMEGA=0.5 SUCC_NUM=200 FAIL_NUM=100 \
    bash robosuite/discriminator/d3disc/scripts/run_d3_benchmark.sh

# Without dynamics ckpt: raw 256-D flow_multi z_t.
WANDB_MODE=offline WANDB_NAME=songgao-personal \
    OMEGA=0.5 SUCC_NUM=200 FAIL_NUM=100 \
    bash robosuite/discriminator/d3disc/scripts/run_d3_benchmark.sh

# Sanity-check omega=0 (reduces to KNN on chosen feature).
OMEGA=0 TASKS="PickPlaceBread" \
    bash robosuite/discriminator/d3disc/scripts/run_d3_benchmark.sh

# Per-task visualizations (4 fail trajectories on PickPlaceBread).
TASK=PickPlaceBread NUM_TRAJS=4 OMEGA=0.5 \
    DYN_CKPT="checkpoints/d3disc/dynamics/d3dyn_v1/d3_dynamics.pt" \
    bash robosuite/discriminator/d3disc/scripts/visualize_d3.sh

# Ablation sweep (omega x k x beta).
bash robosuite/discriminator/d3disc/scripts/ablate_omega_beta_k.sh
```

## Key knobs (env vars exposed by the shell scripts)

### `run_d3_benchmark.sh`

| Env var | Default | Meaning |
|---|---|---|
| `POLICY_CKPT` | `checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_*.pt` | flow_multi checkpoint (frozen encoder) |
| `DYN_CKPT` | `""` | optional trained dynamics predictor; empty = raw z_t feature |
| `CACHE_ROOT` | `data/.lpb_score_cache` | per-demo 256-D latent cache (reused bit-exactly) |
| `OMEGA` | `0.5` | 0 = KNN-only baseline; `{0.1, 0.2, 0.5, 1}` per D1 |
| `K` | `1` | k-NN order, per D5 |
| `BETA`, `KAPPA` | `auto` | F3 slope/offset; `auto` derives from pos self-kNN (1/MAD, median) |
| `SIGMA_SQ` | `0.5` | weighted-KNN bandwidth folding log-weights into distance |
| `DELTA` | `10.0` | tau percentile budget (% false-alarm) |
| `SUCC_NUM`, `FAIL_NUM` | `200`, `100` | per-task trajectory caps at evaluation time |
| `SHARE_BANKS` | `1` | 1 = shared D+/D- across tasks per D8; 0 = per-task banks |

### `train_d3_dynamics.sh`

| Env var | Default | Meaning |
|---|---|---|
| `POLICY_CKPT` | same as above | flow_multi ckpt — only used for cache keys / encoder fallback |
| `TASKS` | 6 canonical envs | used to construct `data/<task>/expert` + `data/<task>/success_rollout` path lists |
| `EXPERT_NUM`, `SUCC_NUM` | `0`, `0` | per-kind trajectory cap at training (0 = all) |
| `HORIZON` | `1` | single-step next-latent prediction (match LPB default) |
| `EPOCHS`, `BATCH_SIZE` | `30`, `256` | |
| `D_MODEL`, `NUM_LAYERS`, `NUM_HEADS` | `512`, `6`, `8` | predictor architecture (LPB defaults) |
| `LR`, `WD` | `3e-4`, `1e-4` | AdamW |
| `EXPERT_RATIO` | `0.5` | weighted sampler mixes expert / success_rollout 50/50 |
| `PROPRIO_LOSS_WEIGHT` | `0.1` | MSE weight for proprio target |
| `SAVE_DIR` | `checkpoints/d3disc/dynamics/<RUN_NAME>` | |

## Notes

- `FlowMultiEncoderWrapper.cache_key` byte-matches the historical cache convention
  (git 5056929). On cache HIT the returned latents are bit-identical; on MISS the
  encoder reproduces cache within ~1e-4 (single-run cuDNN variance across boxes).
- PandaLift has no fail data — it contributes only to `D+` when present.
- Phase 1 is `K=0` step-wise (no dynamics rollout even when using `DYN_CKPT`).
  The trained predictor is used **only** for its projection layers at inference.
  K-step rollout through `f_phi` is a later extension.
- Training the predictor is cheap because the flow_multi encoder is frozen and
  all latents are cached: training is latent -> latent (256-D), no image forward
  per batch. One epoch on the full 6-task pool finishes in a few minutes on a
  single GPU.
