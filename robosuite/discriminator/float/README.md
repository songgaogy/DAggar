# FLOAT in `robosuite/discriminator`

FLOAT (Failure Detection with Optimal Transport) implemented against the shared failure-detector benchmark in `data/utils/benchmark/`, with a pretrained **DINOv2 ViT-B/14** image encoder for per-frame embeddings.

## Algorithm

Given expert trajectories `T_{e,n}`, a rollout prefix `T_b[:t0]`, and a frozen image encoder `phi`:

1. Embeddings
   - `F_{e,n} = {phi(o_{e,i})}` for each expert trajectory
   - `F_b     = {phi(o_{b,j})}` for the current rollout prefix
2. OT cost with uniform marginals `a_i = 1/l_n`, `b_j = 1/t0` and cosine cost `c(i, j) = 1 - cos(phi(o_{e,i}), phi(o_{b,j}))`, solved with an entropic Sinkhorn plan `mu_n^*`: `lambda_n(T_b) = sum_{ij} mu_{n,i,j}^* * c(i, j)`
3. FLOAT index: `lambda(T_b) = min_n lambda_n(T_b)`
4. Threshold calibration over `M` success rollouts: `Lambda = Percentile_{1-delta}({lambda(T_{b,k})})`
5. Failure signal when `lambda(T_b) > Lambda`.

Per-task threshold is calibrated **leave-one-out** over the success bank so a success trajectory never matches itself as an expert (which would trivially collapse `lambda` to ~0 and bias `Lambda` downward).

## Code layout

- `float_core.py` — OT primitives (`sinkhorn`, `cosine_cost_matrix`) and `FLOATComputer`, `ThresholdCalibrator`, `OnlineDetector`. Matches the paper spec directly.
- `float_dino_encoder.py` — `DinoV2ImageEncoder`, a thin wrapper around `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` returning `(T, 768)` CLS embeddings from `(T, H, W, 3) uint8` images.
- `float_benchmark.py` — `FloatBenchmarkDiscriminator`, implements the `data.utils.benchmark.discriminator.Discriminator` protocol. Calibrates one FLOAT computer + threshold per task and caches embeddings per trajectory.
- `float_data.py` / `float_eval.py` — retained for `robosuite.discriminator.lpb.*` which still depends on them. Not used by the FLOAT benchmark path.
- `scripts/run_float_benchmark.sh` — bash entry point that runs the adapter and writes a `benchmark.json` conforming to `data/utils/benchmark/protocol.md`.

## Quick start

```bash
conda activate daggar
bash robosuite/discriminator/float/scripts/run_float_benchmark.sh
```

Output lands at `checkpoints/float/eval/run_<timestamp>/benchmark.json`.

Common overrides (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `TASKS` | `PickPlaceBread PickPlaceCan PickPlaceCereal PickPlaceMilk` | Task filter |
| `FAIL_ROOT` | `data/utils/fail_rollout` | Labeled failure data root |
| `SUCCESS_ROOT` | `data/utils/success_rollout` | Success manifests root |
| `DEVICE` | `cuda` | Torch device |
| `IMAGE_SIZE` | `224` | DINOv2 input size (multiple of 14) |
| `ENCODER_BATCH_SIZE` | `64` | DINOv2 batch size |
| `CAMERA_NAME` | `agentview` | Camera stream used for embeddings |
| `SINKHORN_REG` | `0.05` | Entropic OT regularization |
| `MAX_ITER` | `300` | Sinkhorn iterations |
| `TOL` | `1e-5` | Sinkhorn tolerance |
| `DELTA` | `10.0` | Calibration percentile parameter (%) |
| `STEP_STRIDE` | `8` | `lambda` recompute cadence across rollout steps |
| `MAX_FAIL_PER_TASK` | unset | Cap failure trajectories per task (smoke runs) |
| `MAX_SUCCESS_PER_TASK` | unset | Cap success trajectories per task (smoke runs) |
| `USE_SIMILARITY_COST` | `0` | Set to `1` to use raw cosine similarity as cost |
| `RUN_NAME` | `run_<timestamp>` | Output subdirectory name |

Example smoke run on one task:

```bash
TASKS="PickPlaceCan" \
MAX_FAIL_PER_TASK=2 MAX_SUCCESS_PER_TASK=3 \
bash robosuite/discriminator/float/scripts/run_float_benchmark.sh
```

## Output schema

`benchmark.json` follows `data/utils/benchmark/protocol.md`:

- `discriminator_name == "float_dinov2"`
- `trajectory_level`, `trajectory_level_per_task`
- `step_level`, `step_level_per_task` (failure trajectories only)
- `per_trajectory` — `trajectory_score`, `first_pred_failure_frame`, `first_gt_failure_frame`, `failure_segments`, ...
- `config` — dataset roots, caps, thresholding knobs
- `runtime` — wall time and trajectory counts

The `aux` field on each `DiscriminatorOutput` also exposes the per-task threshold `Lambda`, `delta`, `step_stride`, encoder name, image size, embedding dim, and the index (if any) of the expert skipped due to self-match.

## First run

`torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` downloads ~340 MB into `~/.cache/torch/hub` on the first invocation. Subsequent runs reuse the cache.
