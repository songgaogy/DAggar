# Batch-online DIPOLE pipeline

This package implements one fixed loop:

```text
collect one round -> discriminator -> VAST -> policy -> collect next round
```

The online-update path has been removed. Collection always uses frozen policy and discriminator checkpoints. Training stages run in isolated attempt directories and publish checkpoints only after successful completion.

## Commands

Create a self-contained PickPlaceCereal run and collect round 000:

```bash
bash robosuite/pipeline/scripts/run_online.sh new my_run
```

Train every remaining stage in the active round:

```bash
bash robosuite/pipeline/scripts/train.sh outputs/dipole/PickPlaceCereal/<run> all
```

After policy completion, collect exactly one next round:

```bash
bash robosuite/pipeline/scripts/run_online.sh outputs/dipole/PickPlaceCereal/<run>
```

`train.sh <run_root> disc|vast|policy` runs only the currently legal next stage. It rejects dependency skips.

Evaluate and visualize the discriminator published by one round:

```bash
bash robosuite/pipeline/scripts/eval_disc.sh outputs/dipole/PickPlaceCereal/<run> 0
```

Visualize the VAST checkpoint published by one round:

```bash
bash robosuite/pipeline/scripts/vis_vast.sh outputs/dipole/PickPlaceCereal/<run> 0
```

Both utilities resolve task, checkpoints, CUDA device, and data paths from the run configuration. They are read-only with respect to the workflow state.

## Run contract

```text
<run_root>/
├── config_resolved.yaml
├── manifest.json
├── state.json
├── inputs/
│   ├── checkpoints/
│   └── data/
├── cache/
└── rounds/
    └── 000/
        ├── data/
        ├── disc/
        ├── vast/
        ├── policy/
        └── eval_vis/
            ├── disc/
            │   ├── evaluation/
            │   └── visualization/
            └── vast/
```

All training and collection checkpoints and datasets are copied into `inputs/` with per-file SHA-256 verification. Reflink/COW copies are preferred when supported. Later commands accept only the run root; discriminator evaluation reads its external benchmark dataset through the absolute path saved in `config_resolved.yaml`.

`state.json` is atomically replaced after every transition. Collection keeps complete episodes in `episodes.partial.pt` and atomically publishes `episodes.pt` at the target episode count. A failed training attempt remains under `attempts/`; the next attempt restarts from the last completed dependency instead of restoring optimizer state.

## Configuration

The only public configuration is:

- `config/overall.yaml`
- `config/task/PickPlaceCereal.yaml`

The task file owns immutable initial paths, `history_mix_beta`, and the per-task discriminator, VAST, and policy configuration. `overall.yaml` owns collection, CUDA, environment, storage, and TensorBoard settings shared across tasks. Internal adapters translate this public schema for retained model implementations.

Round 000 warm-starts discriminator, VAST, and policy model weights from the snapshotted initial inputs. Later rounds warm-start each model from the preceding round. Every stage creates a fresh optimizer and scheduler. Policy normalizers are loaded from the parent policy checkpoint and are not refit when cumulative data grows.

## Data and caching

Discriminator nnPU replay always uses the original snapshotted pretrain P/U pools. Online GT-positive and GT-failure pools use the configured recursive round mixture. If the current round has no samples for one GT risk, only that loss is disabled for the round; its configured/effective weights and reason are recorded.

VAST consumes the initial warmup transitions plus policy-BC transitions from every completed round. Policy training consumes expert data plus all routed online streams. Episode and round boundaries remain intact, so sampled windows never cross them.

Frozen discriminator features are cached by a contract hash that includes data, encoder, normalizer, camera, feature-layer, frameskip/action-horizon, image, and GT-window semantics. Mutable head outputs, VAST rewards, and advantages are always recomputed on CUDA. TensorBoard is the only training logger.

## Utilities

The package is organized by responsibility:

- `modules/training/` contains discriminator, VAST, and policy stage implementations.
- `modules/evaluation/` contains discriminator and policy evaluation code.
- `modules/visualization/` contains discriminator and VAST renderers.
- `workflow/` owns the run layout, state transitions, and multi-round runner.
- `common/` owns shared environment and collected-episode contracts.

The standalone VAST warmup utility uses `overall.yaml`:

```bash
bash robosuite/pipeline/scripts/init_vast.sh <output_checkpoint>
```

Rollouts, training, evaluation, and data generation are experiments and should only be started with an explicitly reviewed experiment specification.
