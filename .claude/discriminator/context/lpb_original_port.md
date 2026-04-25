# LPB-Original KNN OOD Discriminator — Port Notes

Date: 2026-04-25
Branch: `discriminator-baseline`
Working dir: `/home/dodo/Documents/DAggar/robosuite`

## Goal

Create a brand-new package `robosuite/discriminator/lpb_original/` that **faithfully
ports the original LPB KNN OOD discrimination logic** from upstream
`tmp/lpb-main/dyn_model/planner_libero.py` into the user's robosuite benchmark.

The user already has another adapted version under `robosuite/discriminator/lpb/`
(different feature/score design). That version was **NOT** to be modified.
An earlier attempt that edited `lpb/` was reverted via `git checkout`.

Discard from upstream LPB:
- diffusion-policy training code
- policy steering / test-time optimization
- env runner / online evaluation

Keep:
- The KNN OOD scoring path (`compute_nn_reward`)
- The dynamics-model training stack so the user can train a fresh checkpoint

## Original-LPB KNN logic (the ground truth being ported)

`tmp/lpb-main/dyn_model/planner_libero.py:147-192`
- Feature per timestep:
  `feat = [encoder(o_t)  ;  proprio_encoder(s_t)]`
  Then per-dim weighting: `[1.0 * visual_dim ;  2.0 * proprio_dim]`.
  No action, no L2-normalize, no projection.
- Distance: non-squared L2 via `torch.cdist(..., p=2)`, k=1, score = `-min_dist`.
- Bank: built once offline from success-demo encoder outputs (`get_demo_latents`).
- The original returns a continuous reward; for our benchmark we add a
  percentile-calibrated threshold to emit binary preds:
  `tau = np.percentile(calibration_min_dists, 100 - delta)`,
  `pred_t = 1 if min_dist_t >= tau`.

## User-confirmed scope (via AskUserQuestion)

| Question | Answer |
|---|---|
| dyn_model dependencies | Full vendor of upstream `dyn_model` |
| Training scripts | Yes — also include dynamics-training entry |
| Benchmark API | Match existing `lpb/lpb_benchmark.py` (BenchmarkTrajectory + DiscriminatorOutput) |
| `diffusion_policy` handling | Copy entire `tmp/lpb-main/diffusion_policy` |
| Training data | Adapt to user's HDF5 (parallel to `lpb/dataset.py`) |
| Hydra | Keep Hydra config |

## Final layout

```
robosuite/discriminator/lpb_original/
├── __init__.py                   # injects sys.path so `import dyn_model` and
│                                 #   `import diffusion_policy` resolve to vendored copies
├── dyn_model/                    # verbatim copy of tmp/lpb-main/dyn_model (~352 KB)
├── _vendored/diffusion_policy/   # tmp/lpb-main/diffusion_policy (1.2 MB after pruning)
├── datasets/
│   ├── __init__.py
│   └── hdf5_dynamics_dataset.py  # HDF5DynamicsModelDataset — API parity with
│                                 #   RobomimicImageDynamicsModelDataset
├── discriminator.py              # KNN OOD core (LPBOriginalEncoder, LPBOriginalKNN,
│                                 #   knn_min_l2_dist, DetectionResult)
├── lpb_benchmark.py              # LPBOriginalBenchmarkDiscriminator: fit_on_benchmark
│                                 #   + score_trajectory matching DiscriminatorOutput
├── train.py                      # Single-GPU Hydra @main; saves checkpoint layout
│                                 #   compatible with dyn_model.plan.load_model
├── conf/                         # Copied from dyn_model/conf, plus:
│   ├── env/hdf5.yaml             # New env config wired to HDF5DynamicsModelDataset
│   └── train_hdf5.yaml           # New training entry config
└── scripts/
    ├── run_lpb_original_benchmark.sh
    └── train_lpb_original_dynamics.sh

data/utils/benchmark/examples/run_lpb_original.py     # CLI runner (parallel to run_lpb.py)
```

### `_vendored/diffusion_policy` pruning (executive call)

Upstream `diffusion_policy` was 114 MB; 112 MB of that was
`env/libero/assets/` (libero simulation OBJ/texture files) which is unused by
the KNN scoring + dynamics-training paths. Pruned subdirs:
`env/`, `real_world/`, `gym_util/`, `env_runner/`, `policy/`, `scripts/`,
`shared_memory/`. Final size: 1.2 MB. Kept: `codecs/`, `common/`, `config/`,
`dataset/`, `model/`, `workspace/`. Restore from tmp/lpb-main if libero
sim ever needed.

## Key implementation details

### `discriminator.py`
- **`LPBOriginalEncoder`** — loads dynamics model via `dyn_model.plan.load_model`
  using Hydra config saved at `<ckpt_dir>/hydra.yaml` and the
  `LinearNormalizer` saved at `<ckpt_dir>/normalizer.pth`. Provides
  `encode_batch(images_per_view, proprio) -> (B, visual_dim + proprio_dim)`
  by running per-view image normalize + crop transform + `model.encode_obs`.
- **`LPBOriginalKNN`** — per-task detector with the original LPB feature
  weighting (visual=1.0, proprio=2.0). `fit(expert, calib)` builds the bank,
  computes calibration min-dists, calibrates `tau = percentile(values, 100 - delta)`.
  `score(features) -> DetectionResult{step_scores, thresholds, preds}`.
- **`knn_min_l2_dist`** — chunked cdist + min, non-squared L2, returns `(B,)`.

### `lpb_benchmark.py`
- `LPBOriginalBenchmarkDiscriminator` mirrors the public interface of
  `lpb/lpb_benchmark.py` (`fit_on_benchmark`, `score_trajectory`, `calibration_summary`,
  `close`).
- Per-task: stratify success demos, split (1 - `calib_fraction`) bank vs.
  `calib_fraction` calibration, encode all, fit detector, store.
- Image pipeline: load BenchmarkTrajectory images by camera, map cameras to
  encoder view names via `camera_to_view` (default identity). Resize to
  `original_img_size` if needed; convert to `[0, 1]` float CHW.
- Proprio: optional `proprio_indices` slice; auto-pad/truncate to
  `model.proprio_encoder.in_chans`.
- Caches encoded trajectories by `(file_path, demo_path)`.

### `datasets/hdf5_dynamics_dataset.py`
- Reads HDF5 layout `demos/<key>/{states, actions, observations/<view>/images}`
  (same schema as existing `lpb/dataset.py`).
- Returns 3-tuples `(obs, act, state)` where:
  - `obs['visual'][view]`: `(num_frames, 3, H, W)` float in `[0, 1]`
  - `obs['proprio']`: `(num_frames, proprio_dim)`
  - `act`: `(num_frames * frameskip, action_dim)` (with the upstream trailing-clip quirk)
- Exposes `proprio_dim`, `action_dim`, `transform`, `original_img_size`,
  `get_normalizer()` — full API parity with upstream zarr datasets.
- Materializes all demos into in-memory numpy arrays (matches upstream
  zarr-into-memory pattern).

### `train.py`
- Hydra `@main(config_name="train_hdf5")`.
- Single-GPU torch loop (no `accelerate` — daggar env doesn't have it).
- Builds model the same way as `dyn_model/train.py:init_models`:
  `ResNetEncoder` + Hydra-instantiated proprio/action encoders + ViT predictor +
  `VisualDynamicsModel`.
- Saves `<run_dir>/{checkpoints/model_<epoch>.pth, hydra.yaml, normalizer.pth}`
  in the layout `dyn_model.plan.load_model` expects, so the discriminator can
  load the checkpoint as-is.
- Dependency: `policy_ckpt_path` (a pretrained diffusion-policy checkpoint with
  `obs_encoder.obs_nets[view].backbone`) is required for `ResNetEncoder` to
  bootstrap the visual backbone — this is an upstream-LPB hard requirement.

## Verifications done

- All top-level modules import cleanly in the daggar conda env.
- `dyn_model` and `diffusion_policy` resolve to the vendored paths inside
  `lpb_original/`.
- `knn_min_l2_dist` matches `torch.cdist(...).min(dim=1).values`.
- `LPBOriginalKNN.fit` + `score` round-trip on synthetic features (no
  encoder needed) — threshold == `np.percentile(calib, 90)` for `delta=10`.
- `run_lpb_original.py --help` lists the full CLI.
- Both bash scripts pass `bash -n` syntax check.
- **Not run**: end-to-end with a real checkpoint (none available; requires
  user to run `train_lpb_original_dynamics.sh` first).

## How to run

### Train dynamics (requires DP policy ckpt for ResNet bootstrap)
```bash
TRAIN_DATA=/abs/path/data/utils/success_rollout/PickPlaceCan \
POLICY_CKPT=/abs/path/dp_policy.ckpt \
ACTION_DIM=7 PROPRIO_DIM=9 \
bash robosuite/discriminator/lpb_original/scripts/train_lpb_original_dynamics.sh
```

### Run KNN benchmark
```bash
MODEL_CKPT=/abs/path/checkpoints/lpb_original/dynamics/.../checkpoints/model_50.pth \
TASKS="PickPlaceCan" MAX_FAIL_PER_TASK=10 MAX_SUCCESS_PER_TASK=20 \
bash robosuite/discriminator/lpb_original/scripts/run_lpb_original_benchmark.sh
```

## Caveats / future work

1. `accelerate` is missing in daggar env. Vendored `dyn_model/train.py` won't
   run as-is; use `lpb_original/train.py` (single-GPU, no accelerate).
2. `policy_ckpt_path` (diffusion-policy ckpt with `obs_encoder`) is a hard
   upstream requirement. If user doesn't have one, `ResNetEncoder` would need
   to be modified to fall back to ImageNet-pretrained ResNet18 — out of scope.
3. HDF5 schema is fixed: `demos/<key>/{states, actions, observations/<view>/images}`.
   Camera/view name mismatches between training data and config can be papered
   over via `--camera-to-view` at benchmark time.
4. Language encoder path (libero text features) is wired through the loader
   only when `'libero' in cfg.env.train_data_path`. With the new `hdf5` env
   config it stays disabled — fine for the user's robosuite tasks.
5. End-to-end smoke run on real data is the user's next step; intermediate
   sanity checks all pass.

## Files touched in this conversation (final state)

Created:
- `robosuite/discriminator/lpb_original/__init__.py`
- `robosuite/discriminator/lpb_original/dyn_model/...` (verbatim copy)
- `robosuite/discriminator/lpb_original/_vendored/diffusion_policy/...` (pruned copy)
- `robosuite/discriminator/lpb_original/datasets/__init__.py`
- `robosuite/discriminator/lpb_original/datasets/hdf5_dynamics_dataset.py`
- `robosuite/discriminator/lpb_original/discriminator.py`
- `robosuite/discriminator/lpb_original/lpb_benchmark.py`
- `robosuite/discriminator/lpb_original/train.py`
- `robosuite/discriminator/lpb_original/conf/...` (copied + new `env/hdf5.yaml`,
  `train_hdf5.yaml`)
- `robosuite/discriminator/lpb_original/scripts/run_lpb_original_benchmark.sh`
- `robosuite/discriminator/lpb_original/scripts/train_lpb_original_dynamics.sh`
- `data/utils/benchmark/examples/run_lpb_original.py`

Reverted (initial misdirected work):
- `robosuite/discriminator/lpb/knn_discriminator.py` (`git checkout`)
- `robosuite/discriminator/lpb/lpb_benchmark.py` (`git checkout`)
- `robosuite/discriminator/lpb/scripts/run_lpb_benchmark.sh` (`git checkout`)
- `data/utils/benchmark/examples/run_lpb.py` (manual revert; not git-tracked)
