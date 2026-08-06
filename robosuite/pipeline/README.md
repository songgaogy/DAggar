# Robosuite Pipeline

End-to-end training and evaluation pipeline for **Flow-DAgger**: online imitation learning with a flow-matching visuomotor policy in [robosuite](https://robosuite.ai/) environments. The pipeline supports human-in-the-loop (HIL) intervention via SpaceMouse, offline pretraining from HDF5 demonstrations, and post-hoc supervised fine-tuning (SFT) on collected rollout data.

## Overview

The pipeline trains a **multi-modal flow policy** (`flow-multi`) that conditions on:

- Multi-camera RGB images (e.g. `agentview`, `robot0_robotview`, `robot0_eye_in_hand`)
- Proprioceptive state
- Natural-language task instructions (CLIP text encoder)

Actions are predicted as **action chunks** (default horizon = 8) via flow matching and executed in the simulator. Training follows a **DAgger-style** loop: the policy rolls out in the environment, human corrections are recorded, and both online and demo replay buffers are updated for continual learning.

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  HDF5 Demos     │────▶│  Demo Buffer     │────▶│                     │
│  (pretrain)     │     │                  │     │  FlowDaggerPolicy   │
└─────────────────┘     └──────────────────┘     │  (flow matching)    │
                                                 │                     │
┌─────────────────┐     ┌──────────────────┐     │  ResNet18 + CLIP    │
│  Robosuite Env  │◀───▶│  Online Buffer   │────▶│  + Transformer      │
│  + SpaceMouse   │     │  (interventions) │     │  + 1D UNet head     │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
```

## Directory Structure

```
pipeline/
├── config/                  # Hydra YAML configs
│   ├── train_flow_dagger.yaml   # Main Flow-DAgger training config (composes discriminator)
│   ├── sft.yaml                 # Offline SFT config
│   └── discriminator.yaml       # PU-BCE failure-discriminator runtime config (cfg.discriminator)
├── flow_dagger/             # Core algorithm
│   ├── agent.py                 # FlowDaggerAgent: buffers + policy wrapper
│   ├── trainer.py               # FlowDaggerTrainer: update scheduling, async learner
│   ├── replay_buffer.py         # Chunked replay with image augmentation
│   ├── common.py                # Config dataclasses
│   ├── discriminator.py         # DiscriminatorRuntime: background fail/safe indicator + HUD
│   └── models/flow.py           # FlowDaggerPolicy: training + ODE inference
├── envs/                    # Robosuite environment integration
│   └── robosuite.py             # Env builder, viewer, SpaceMouse intervention
├── sft/                     # Offline fine-tuning scripts
│   ├── flow_sft.py              # SFT from HDF5 demo splits
│   ├── dagger_sft.py            # SFT from saved DAgger demo chunks
│   └── eval_flow_offline.py     # Offline checkpoint evaluation
├── scripts/                 # Bash entry points
│   ├── train_flow_dagger.sh
│   └── eval_flow_dagger.sh
├── utils/                   # I/O, logging, determinism, training helpers
│   ├── io.py                    # HDF5 / transition-shard serialization
│   ├── determinism.py           # Seeded env resets, disjoint-seed assertions
│   ├── train_utils.py           # Shared training helpers (rate limiters, loggers, checkpoint paths)
│   └── online.py                # Flow-DAgger online-rollout helpers (obs builders, HDF5 demo loader, run/device resolution, async demo-chunk writer) extracted from train_flow_dagger.py
├── common/                  # Shared types (Transition, EncoderConfig, etc.)
├── factory.py               # Algorithm registry (`build_algorithm`)
├── train_flow_dagger.py     # Main online training entry point (`main()` control loop; helpers live in utils/online.py)
└── eval_flow_dagger.py      # Checkpoint evaluation entry point
```

## Key Components

### FlowDaggerAgent (`flow_dagger/agent.py`)

Central agent that owns:

- **FlowDaggerPolicy** — the trainable flow-matching model
- **online_buffer** — transitions from live rollouts (policy + interventions)
- **demo_buffer** — transitions from offline HDF5 demos and intervention segments

Built via `build_algorithm(cfg)` from `factory.py` (registered as `"flow-dagger"`).

### FlowDaggerTrainer (`flow_dagger/trainer.py`)

Manages the training loop:

- Bootstraps the demo buffer from offline HDF5 data
- Records transitions with intervention labels (`is_intervention`)
- Schedules gradient updates (sync or async with backpressure via `max_pending_updates`)
- Supports pretrain warmup (`pretrain_steps`) before online rollout

### Robosuite Integration (`envs/robosuite.py`)

- **`build_robosuite_env`** — constructs robosuite environments from `RobosuiteRuntimeConfig`
- **`RobosuiteInterventionRuntime`** — SpaceMouse override: when the human moves the device or toggles the gripper, the policy action is replaced with the human command
- **`RobosuiteViewerRuntime`** — decoupled viewer for interactive training
- Sparse success reward via `sparse_success_reward`

### Failure Discriminator (`flow_dagger/discriminator.py`)

A **PU-BCE failure discriminator** (frozen DINOv3 dynamics encoder + nnPU head, from
`robosuite.discriminator.dyn_disc`) runs alongside online rollout as a **human-intervention
indicator**. It has **zero effect on policy updates or learning** — it only displays
`fail`/`safe` and (optionally) pauses the env to prompt a human.

- **`DiscriminatorRuntime`** — loads the encoder + head once from a single
  `pu_bce_head.pth` checkpoint (the frozen dynamics `encoder_ckpt` and head dims are read
  from the checkpoint payload, overridable in config). A **background daemon thread** runs
  the heavy DINOv3 encode + scoring on the latest published frame, so the control loop is
  never blocked.
- **Scoring cadence (`inference.fps`)** — two modes:
  - **`fps: -1` (default) — chunk-triggered.** The discriminator scores **exactly once per
    new policy action chunk**, i.e. every time the policy infers a fresh chunk (every
    `execute_horizon` control steps). Scoring is locked to the policy's planning rhythm
    rather than wall-clock time.
  - **`fps > 0` — legacy time-gated.** The discriminator scores at a fixed `fps` cadence,
    decoupled from the policy's chunk boundaries (the previous behaviour).
- **Snapshot model** — the **main thread** owns the MuJoCo sim and `publish(...)`es each
  scored frame (the two training views `agentview` + `robot0_eye_in_hand` rendered at
  256×256, Panda `qpos[7]+qvel[7]` proprio selected via the dynamics `proprio_map`, and the
  policy's **planned future action chunk**). The worker thread only reads that snapshot —
  it never touches the sim.
- **Always the policy action, even under intervention** — the action fed to the
  discriminator is **always the policy's inferred action chunk**, never the human override.
  While a human is intervening, the policy keeps inferring in the background (at its normal
  chunk cadence, on live observations) purely to supply this chunk — the inferred action is
  **not executed** (the human command still drives the env). Background policy inference
  during intervention runs only when the discriminator is enabled.
- **Pause / resume** (only when `inference.intervene_env=true`) — once the policy is judged
  `fail` for `pause.consecutive_fail_frames` consecutive scored frames, the env pauses and
  waits. Press **ENTER** to resume policy rollout, or **intervene with the SpaceMouse**
  (recorded to the demo buffer as usual) — either resumes. After a resume it re-arms only
  once the discriminator returns to `safe` and fails again (debounce), so it won't repeatedly
  interrupt. The async learner keeps training throughout.
- **Console** — when the discriminator is enabled, the verbose rollout prints are redirected
  to `console.log` and the terminal shows a single-line, in-place HUD: `fail` (red) / `safe`
  (green) / `PAUSED` (yellow) plus `ep`, `step`, `pending`, and the failure score.
- **`build_discriminator_runtime(cfg.discriminator)`** returns `None` when `enabled=false`,
  and a load failure is caught and logged without aborting training.

### Policy Architecture

Configured under `algorithm.flow.model` in `train_flow_dagger.yaml`:

| Module | Default |
|--------|---------|
| Image encoder | Shared ResNet-18 + spatial softmax |
| Proprio encoder | MLP |
| Language encoder | CLIP ViT-B/32 (frozen) |
| Modulation | FiLM |
| Fusion | Transformer (3 layers, 8 heads) |
| Action head | 1D UNet with cross-attention |

Inference integrates a learned velocity field over `n_ode_steps` Euler steps to produce action sequences.

## Workflows

### 1. Flow-DAgger Online Training

Interactive training with optional human intervention and async learner updates.

```bash
# From repo root, using the provided script
bash robosuite/pipeline/scripts/train_flow_dagger.sh

# Or directly via Hydra
python -m robosuite.pipeline.train_flow_dagger \
  env.environment=PickPlaceCereal \
  data.num_trajectories=10 \
  runtime.init_checkpoint=checkpoints/multitask_6/flow_multi_ep0100.pt \
  intervention.enabled=true
```

**Training loop (simplified):**

1. Load init checkpoint and optionally bootstrap demo buffer from HDF5 pretrain data
2. Run `pretrain_steps` of offline updates on the demo buffer
3. Roll out policy in robosuite; human can intervene via SpaceMouse
4. Store transitions; intervention segments go to the demo buffer
5. Async/sync gradient updates on mixed batches from both buffers
6. Save checkpoints every configured number of completed online episodes, plus demo buffer chunks and WandB logs

Key runtime flags:

| Flag | Description |
|------|-------------|
| `intervention.enabled` | Enable SpaceMouse HIL override |
| `runtime.async_updates` | Run learner in background thread |
| `runtime.online_updates_enabled` | Enable/disable online gradient updates |
| `runtime.save_demo_buffer` | Persist intervention demo chunks to disk |
| `runtime.viewer_async` | Refresh the decoupled viewer asynchronously; the training shell defaults to synchronous refresh |
| `algorithm.trainer.max_pending_updates` | Cap async learner lag (default: 8) |
| `logging.checkpoint_episode_interval` | Save a checkpoint every N completed online episodes (default: 20) |
| `discriminator.enabled` | Run the PU-BCE fail/safe indicator thread + HUD |
| `discriminator.inference.intervene_env` | Pause the env on sustained `fail` and wait for the human |

Outputs are written to `logging.output_root/<run_name>/` with:

- `checkpoints/` — periodic model checkpoints (`ep_*.pt`) and `latest.pt`
- `buffers/demo_chunks/` — saved intervention transitions
- `console.log`, `metrics_runtime.jsonl` — runtime logs
- `resolved_config.yaml`, `run_info.json` — reproducibility metadata

`scripts/train_flow_dagger.sh` defaults to synchronous viewer refresh to keep each displayed frame aligned
with the latest completed environment step. Set `VIEWER_ASYNC=true` to restore the asynchronous preview.
Runtime logs include `viewer_fps` and `policy_inference_ms_mean/max/count` to distinguish viewer slowdown
from the synchronous action-chunk inference latency.

#### Failure-discriminator indicator

The PU-BCE discriminator (see [Failure Discriminator](#failure-discriminator-flow_daggerdiscriminatorpy))
is composed into the run via Hydra `defaults` as `cfg.discriminator`. It is **on by default**
(`discriminator.enabled=true`) and acts purely as an intervention indicator.

The `scripts/train_flow_dagger.sh` entrypoint overrides this to disabled by default. Set
`DISCRIMINATOR_ENABLED=true` to enable it, and use `CHECKPOINT_EPISODE_INTERVAL` to override
the default 20-episode checkpoint interval:

```bash
DISCRIMINATOR_ENABLED=true CHECKPOINT_EPISODE_INTERVAL=10 \
  bash robosuite/pipeline/scripts/train_flow_dagger.sh
```

```bash
# Indicator only — HUD shows fail/safe, never pauses (no effect on learning):
python -m robosuite.pipeline.train_flow_dagger \
  env.environment=PickPlaceCereal \
  discriminator.inference.intervene_env=false

# Pause-on-fail — env pauses after sustained fail; ENTER or SpaceMouse intervention resumes:
python -m robosuite.pipeline.train_flow_dagger \
  env.environment=PickPlaceCereal \
  intervention.enabled=true \
  discriminator.inference.intervene_env=true

# Disable entirely:
python -m robosuite.pipeline.train_flow_dagger discriminator.enabled=false
```

> `discriminator.task_name` must match the env / the dynamics model's `proprio_map` tasks
> (per-task thresholds are calibrated). The `checkpoint` (`pu_bce_head.pth`) embeds the
> frozen dynamics `encoder_ckpt` path; both are overridable in `config/discriminator.yaml`.

### 2. Offline SFT (HDF5 Demos)

Fine-tune a checkpoint on mixed demo splits without environment interaction.

```bash
bash robosuite/pipeline/sft/finetune_flow.sh

# Or via Python
python -m robosuite.pipeline.sft.flow_sft \
  env.environment=NutAssemblySquare \
  runtime.init_checkpoint=checkpoints/multitask_6/flow_multi_ep0100.pt \
  data.sft_data.expert=10 \
  data.sft_data.pretrain_data=30
```

Supports mixing data from logical splits: `expert`, `pretrain_data`, `success_rollout`, `fail_rollout`. Uses EMA weight averaging and configurable `train_scope` (e.g. `"all"`, `"head_only"`).

### 3. DAgger SFT (Saved Demo Chunks)

Fine-tune on intervention data collected during Flow-DAgger runs, optionally warmed up with pretrain HDF5 demos.

```bash
bash robosuite/pipeline/sft/dagger_sft.sh

python -m robosuite.pipeline.sft.dagger_sft \
  data.demo_buffer=outputs/.../buffers/demo_chunks/chunk_00000000.pt \
  data.warmup_num=10
```

### 4. Evaluation

Evaluate a trained checkpoint with deterministic or stochastic action sampling.

```bash
bash robosuite/pipeline/scripts/eval_flow_dagger.sh

python -m robosuite.pipeline.eval_flow_dagger \
  --checkpoint outputs/.../checkpoints/step_00015000.pt \
  --env-name PickPlaceCereal \
  --task-name PickPlaceCereal \
  --episodes 50 \
  --video-output true
```

Eval seeds default to a high band (`900000+`) disjoint from training seeds to avoid layout overlap.

## Configuration

All configs use [Hydra](https://hydra.cc/). The canonical training config is `config/train_flow_dagger.yaml`.

**Top-level sections:**

| Section | Purpose |
|---------|---------|
| `env` | Robosuite task, robots, cameras, image size, control frequency |
| `algorithm` | Flow policy, buffers, trainer hyperparameters |
| `intervention` | SpaceMouse device settings |
| `data` | Demo paths, splits, trajectory counts |
| `runtime` | Interactive mode, async updates, checkpoint resume, FPS limits |
| `logging` | WandB project, output directory, save intervals |
| `discriminator` | PU-BCE fail/safe indicator: checkpoint, device, `inference.fps` (`-1` = score per new policy action chunk; `>0` = fixed-rate), `inference.intervene_env`, `pause.*` debounce, `hud.enabled` (composed from `config/discriminator.yaml`) |

Override any field from the CLI:

```bash
python -m robosuite.pipeline.train_flow_dagger \
  algorithm.flow.learning_rate=5e-5 \
  algorithm.trainer.batch_size=32 \
  logging.use_wandb=false
```

## Data Format

Demonstrations are stored as **HDF5** files under `data/<task_name>/<split>/`. Each demo group contains:

- `actions` — per-step action vectors
- `states` — simulator states for proprio extraction
- `observations/<camera>/images` — RGB frames
- `intervention_labels` (optional) — boolean intervention flags

During training, demos are converted to `Transition` objects and loaded into replay buffers. Intervention chunks can also be saved/loaded as PyTorch shards (`.pt`) via `utils/io.py`.

## Dependencies

- **robosuite** — simulation environments
- **PyTorch** — model training and inference
- **Hydra / OmegaConf** — configuration management
- **h5py** — HDF5 demo loading
- **WandB** — experiment logging (default: offline mode)
- **robosuite.policy.flow_multi** — underlying flow policy architecture (`build_flow_policy`)

Recommended conda environment: `dagger`

```bash
/home/dodo/miniconda3/envs/dagger/bin/python -m robosuite.pipeline.train_flow_dagger ...
```

## Public API

```python
from robosuite.pipeline import (
    FlowDaggerAgent,
    FlowDaggerTrainer,
    Transition,
    build_algorithm,
    build_robosuite_env,
    load_demo_paths,
)
```

Algorithm registration is extensible via `@register_algorithm` in `factory.py`.
