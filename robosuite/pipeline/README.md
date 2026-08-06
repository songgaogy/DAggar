# AWR Baseline

This package contains one supported online learning method: Advantage-Weighted
Regression (AWR) with flow action chunks. It targets `PickPlaceCereal` and uses
CUDA for every tensor computation.

## Training behavior

- The policy initializes from the configured flow-multi checkpoint.
- Expert, successful, and failed trajectories initialize the replay data.
- Sparse rewards use `-1` for non-success and `0` for success.
- Q/V critics use discounted action chunks, double Q, expectile value fitting,
  and an optional reusable Q/V warmup cache.
- Every executed transition enters the online replay buffer. Human intervention
  replaces the policy action and is also copied to the demonstration buffer.
- Training runs synchronously for `trainer.updates_per_train=2000` updates
  after every `trainer.episodes_per_train=10` interactive episodes. There is no
  alternative algorithm or asynchronous learner mode.
- Metrics are written to TensorBoard, JSONL, and the console log. WandB is not
  used.

The pipeline rejects CPU devices instead of silently falling back when CUDA is
unavailable.

## Configuration

Hydra composes `config/overall.yaml` with
`config/task/PickPlaceCereal.yaml`. The task file contains only task-specific
environment, demonstration, horizon, discount, and grasp-penalty values.

Outputs are written below:

```text
outputs/baseline/awr/PickPlaceCereal/<run_name>/
```

Each training run contains the resolved config, run metadata, console and JSONL
logs, TensorBoard events, checkpoints, replay-buffer chunks, demonstration
cache. Reusable Q/V caches are stored under
`outputs/baseline/awr/qv_cache/`.

## Commands

Interactive training:

```bash
bash robosuite/pipeline/scripts/train.sh
```

Use one GPU for learner and inference:

```bash
bash robosuite/pipeline/scripts/train.sh \
  runtime.learner_device=cuda:0 \
  runtime.inference_device=cuda:0
```

Resume a new-format run:

```bash
bash robosuite/pipeline/scripts/train.sh \
  checkpoint.resume=true \
  checkpoint.path=/absolute/path/to/latest.pt
```

Build or refresh the Q/V cache:

```bash
bash robosuite/pipeline/scripts/build_qv_cache.sh
```

Headless deterministic evaluation:

```bash
CHECKPOINT=/absolute/path/to/latest.pt \
bash robosuite/pipeline/scripts/eval.sh
```

Save evaluation videos with `evaluation.save_video=true`.

Visualize Q/V values on successful rollout data:

```bash
CHECKPOINT=/absolute/path/to/latest.pt \
bash robosuite/pipeline/scripts/visualize_q.sh
```

All launchers pass additional arguments through as Hydra overrides.

## Package layout

- `src/awr/` contains the AWR agent, trainer, flow policy, critics, and Q/V
  cache support.
- `src/data/` contains transitions, replay persistence, and demonstration
  loading.
- `src/environment/` contains the robosuite adapter and SpaceMouse
  intervention runtime.
- `utils/` contains CUDA, runtime, and logging helpers.
- `train.py`, `eval.py`, `build_qv_cache.py`, and `visualize_q.py` are the
  public entry points.

Checkpoints from the previous `outputs/awr` implementation are intentionally
incompatible. The configured flow-multi initialization checkpoint remains
supported, and checkpoints produced by this layout support resume.
