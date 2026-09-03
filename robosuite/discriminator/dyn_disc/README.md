# RPT Representation Pretraining and nnPU Discriminator

This branch supports one representation pipeline:

```text
robosuite success_rollout HDF5
  -> frozen DINOv3 mean-pooled patch latents
  -> RPT masked sensorimotor pretraining
  -> frame-level RPT action-token features
  -> unchanged nnPU discriminator
```

Historical WAM/future-dynamics models, KNN scoring, Agilex data, and
preprocessed-cache backends are intentionally unsupported. All run artifacts
belong under `checkpoints/dyn_disc/ablations/RPT/`.

## Method

RPT receives eight consecutive sensorimotor steps. Every step contains tokens
in `[agentview, robot0_eye_in_hand, proprio, action]` order. The two visual
tokens share their projection, mask token, and reconstruction head. During
pretraining, learned slot positional embeddings distinguish the fixed camera,
proprio, and action positions while a temporal embedding is shared by all four
tokens at each time step. 70–90% of the 32 tokens are independently masked and the model
reconstructs only masked targets. At least one visible and one masked token are
guaranteed per sample.

The frozen DINOv3 ViT-B/16 runs once during cache generation. CLS/register
tokens are excluded and patch tokens are mean pooled into one float32 768-D
latent per camera and frame. RPT predicts those visual latents together with
normalized 14-D proprioception and 7-D actions.

Downstream encoding is causal: frame `t` uses `[t-7, ..., t]`, left-padded with
frame zero, and returns the final action-token hidden state `(T, 192)`. The
feature source is always `rpt_action_token`.

## Data and outputs

The default seven tasks are:

```text
NutAssemblyRound NutAssemblySquare PickPlaceBread PickPlaceCan
PickPlaceCereal PickPlaceMilk Stack
```

Pretraining reads at most 100 trajectories per task from `success_rollout`.
Default paths are:

```text
checkpoints/dyn_disc/ablations/RPT/
  cache/dinov3_mean_v1/
  pretrain/rpt_robosuite-<timestamp>/
    hydra.yaml
    normalizer.pth
    tensorboard/
    checkpoint/model_{10,20,30,40,50}.pth
  nnpu/run_<timestamp>_<tasks>/
  visualizations/video/<task>-<timestamp>/
  visualizations/latent/<task>-<timestamp>/
```

Cache manifests fingerprint source files, DINO settings, view ordering,
proprio mappings, dimensions, and shard metadata. Cache or checkpoint metadata
mismatches fail loudly; legacy dynamics checkpoints are not accepted.

## Cache and pretrain RPT

Use the repository environment and two visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  bash robosuite/discriminator/dyn_disc/scripts/train_rpt_robosuite.sh
```

The launcher first builds/reuses the sharded DINO cache, then starts two-rank
DDP RPT training. Tensor operations are CUDA-only. Defaults are global batch
4096, 50 epochs, fused AdamW, peak LR `4e-4`, 8-epoch linear warmup, cosine
decay, BF16 autocast, and checkpointing every 10 epochs. No formal training is
started merely by importing this package.

Important environment overrides include `DATA_ROOT`, `CACHE_ROOT`,
`DINO_MODEL_PATH`, `TASKS`, `CACHE_BATCH_SIZE`, `NPROC_PER_NODE`, and
`OUTPUT_ROOT`, `PRETRAIN_OUT_DIR`, and `PYTHON_BIN`. Additional Hydra overrides
may be appended to the command.

Cache loading uses independent HDF5 worker processes, pinned-memory prefetch,
native time-chunk-aligned reads, and a combined two-camera DINO batch. On the
default mechanical disk, start with `CACHE_NUM_WORKERS=2` per GPU and tune
`1/2/4`; more workers can reduce throughput through disk seeking. The default
`CACHE_PREFETCH_FACTOR=2` bounds host memory use. `CACHE_BATCH_SIZE=800` means
at most 800 total camera images per DINO forward; for the current 400-frame
demos this combines both views into one forward while remaining aligned to the
native 50-frame gzip chunk. Cache logs report aggregate images/s for each shard.

## Train and evaluate nnPU

nnPU treats success frames as positive and entire failure rollouts as
unlabeled; GT failure timing is never used for training. The non-negative PU
risk, balanced sampling, fixed-epoch fit, success-percentile calibration, and
`failure_score = -g(z)` semantics are unchanged.

```bash
MODEL_CKPT=/abs/path/to/checkpoint/model_50.pth \
TASKS="PickPlaceCereal PickPlaceMilk" \
  bash robosuite/discriminator/dyn_disc/scripts/run_pu_bce_robosuite_benchmark.sh
```

The runner writes `benchmark.json`, `unlabeled_pool_manifest.json`, and
`checkpoints/pu_bce_head.pth`. The nnPU checkpoint records
`pretraining_method=rpt` and the exact RPT representation fingerprint.

## Visualizations

Render score videos and a PDF from an already fitted head:

```bash
MODEL_CKPT=/abs/path/to/checkpoint/model_50.pth \
LOAD_CKPT=/abs/path/to/pu_bce_head.pth \
TASK=PickPlaceCereal \
  bash robosuite/discriminator/dyn_disc/scripts/visualize_pu_bce_robosuite.sh
```

Load-only mode discovers only evaluation trajectories; it does not inspect or
require an unlabeled training pool. `NO_FLIP_VERTICAL=1` forwards the frame
orientation option. The head and representation fingerprints must match.

Generate PCA/t-SNE plots and export the underlying frame features:

```bash
MODEL_CKPT=/abs/path/to/checkpoint/model_50.pth \
TASK=PickPlaceCereal \
  bash robosuite/discriminator/dyn_disc/scripts/visualize_rpt_latents.sh
```

GT masks only color the latent plot as `success`, `failure-before-GT`, and
`failure-after-GT`; they never enter RPT or nnPU training.

## Environment

Launchers default to `/home/dodo/miniconda3/envs/dagger/bin/python`, require
CUDA, and never fall back to CPU for tensor computation. TensorBoard is the
only training logger.
