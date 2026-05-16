# robosuite.pipeline

This directory contains online robosuite training, evaluation, rendering, and
human-intervention entry points. The algorithms are intentionally thin wrappers
around a shared runtime stack: robosuite env construction, image observation
rendering, GUI preview, SpaceMouse / keyboard input, async training, async
checkpointing, and replay-buffer persistence.

## Main Entry Points

Use the helper scripts unless you need low-level Hydra overrides:

```bash
# HIL-SERL online training with optional intervention.
bash robosuite/pipeline/scripts/train_hil_serl.sh

# Flow-DAgger training or frozen-policy eval.
FLOW_DAGGER_MODE=train bash robosuite/pipeline/scripts/train_flow_dagger.sh
FLOW_DAGGER_MODE=eval bash robosuite/pipeline/scripts/train_flow_dagger.sh

# AWR training.
bash robosuite/pipeline/scripts/train_awr.sh

# HG-DAgger training.
bash robosuite/pipeline/scripts/train_hg_dagger.sh

# Collect intervention demos into HDF5.
python -m robosuite.pipeline.collect_human_intervention
```

The scripts assume the repo root is `$HOME/Documents/DAggar/robosuite` by
default. Override `ROOT_DIR` if needed.

The expected Python environment is `daggar`; `train_awr.sh` uses
`/home/dodo/miniconda3/envs/daggar/bin/python` by default. For other direct
module commands, run them from the repo root after activating the same env.

## GUI / Render Startup

The two flags that decide whether a desktop window is created are:

- `runtime.interactive=true`
- `runtime.viewer_enabled=true`

The scripts also force:

```bash
env.renderer=mjviewer
```

When `INTERACTIVE=true`, the scripts export:

```bash
MUJOCO_GL=glfw
```

and require `$DISPLAY` to be non-empty. If no desktop X session is available,
run headless:

```bash
INTERACTIVE=false VIEWER_ENABLED=false INTERVENTION_ENABLED=false \
bash robosuite/pipeline/scripts/train_hil_serl.sh
```

Headless mode exports:

```bash
MUJOCO_GL=egl
```

Human intervention also requires a GUI session because robosuite input devices
import desktop input hooks:

```bash
INTERVENTION_ENABLED=true
```

requires `$DISPLAY`, even if the viewer itself is disabled.

## Render Architecture

The pipeline separates three different rendering concerns:

1. Rollout simulation: the env that receives actions and advances physics.
2. Policy image observations: offscreen camera images used by the agent.
3. Human GUI preview: a desktop window for teleoperation or visual debugging.

The shared pieces live in:

- `envs/robosuite.py`
  - `RobosuiteRuntimeConfig`
  - `build_robosuite_env`
  - `RobosuiteObservationAdapter`
  - `RobosuiteViewerRuntime`
  - `snapshot_env_state` / `apply_snapshot_to_env`
- `utils/train_utils.py`
  - `build_runtime_cfg`
  - `resolve_camera_names`
  - `resolve_render_camera`
  - `reset_observation_adapter`

### HIL-SERL and HG-DAgger

`train_hil_serl.py` and `train_hg_dagger.py` use this layout:

- `main_env`
  - owns the rollout simulation
  - opens the native `mjviewer` window when `runtime.interactive=true` and
    `runtime.viewer_enabled=true`
  - is created with `has_renderer=main_has_renderer`
  - is created with `has_offscreen_renderer=false`
- `obs_render_env`
  - is always headless
  - is created with `has_renderer=false`
  - is created with `has_offscreen_renderer=true` when policy cameras exist
  - renders camera images for policy observations

`RobosuiteObservationAdapter` bridges them. Before rendering policy images, it
copies the MuJoCo state from `main_env` into `obs_render_env` through
`snapshot_env_state(...)` and `apply_snapshot_to_env(...)`, then calls
`render_env.sim.render(...)` for each policy camera.

This design keeps the user-facing `mjviewer` separate from policy image
generation. Resets use `reset_observation_adapter(..., preserve_mjviewer=True)`
when the main env owns a window, so the viewer is not destroyed on every
episode reset.

### Flow-DAgger and AWR

`train_flow_dagger.py` and `train_awr.py` use policy camera observations from
the rollout env, and can create a decoupled GUI mirror through
`RobosuiteViewerRuntime`.

When the decoupled viewer is enabled:

1. The rollout env advances physics and serves policy observations.
2. A separate `viewer_env` is built only for display.
3. The rollout loop calls `viewer_runtime.publish_from_env(env)`.
4. The viewer runtime copies the latest snapshot into `viewer_env`.
5. The viewer refreshes at `runtime.render_fps`.

Important viewer controls:

```bash
VIEWER_BACKEND=mjviewer   # native robosuite window, script default
VIEWER_BACKEND=opencv     # accepted runtime backend label; forced sync
VIEWER_BACKEND=auto       # choose based on renderer/camera setup
VIEWER_ASYNC=true         # background thread for mjviewer refresh
RENDER_FPS=10             # display refresh cap
```

In synchronous viewer mode, the rollout loop calls `render_if_due()` and pays
the viewer cost inside the actor loop. In async `mjviewer` mode,
`RobosuiteViewerRuntime.start()` launches a daemon thread named
`hil_serl_gui`; the rollout loop only publishes the newest snapshot.

The `opencv` backend label is forced to synchronous mode in
`RobosuiteViewerRuntime` because it is not treated as thread-owned by the
current implementation.

## Creating a GUI Window

Minimal GUI training command:

```bash
INTERACTIVE=true VIEWER_ENABLED=true INTERVENTION_ENABLED=true \
bash robosuite/pipeline/scripts/train_hil_serl.sh
```

Minimal GUI eval command for Flow-DAgger:

```bash
FLOW_DAGGER_MODE=eval INTERACTIVE=true VIEWER_ENABLED=true \
VIEWER_ASYNC=false VIEWER_BACKEND=mjviewer \
bash robosuite/pipeline/scripts/train_flow_dagger.sh
```

If the window does not appear, check these first:

- Run from a GUI terminal or VSCode session with a valid `$DISPLAY`.
- Keep `MUJOCO_GL=glfw` for interactive rendering.
- Use `env.renderer=mjviewer`.
- Disable headless flags: `INTERACTIVE=true VIEWER_ENABLED=true`.
- For intervention, keep `INTERVENTION_ENABLED=true` and use a supported device
  config.

## Policy Image Cameras vs Viewer Camera

Policy cameras and the GUI camera are intentionally independent:

- `env.camera_names`: cameras rendered into policy observations.
- `env.render_camera`: camera shown by the GUI viewer.

If `env.render_camera` is not set, the code falls back to the first policy
camera. Use this when you want the policy to train from one view but watch a
different view in the GUI.

## Runtime Frequencies

The actor loop has separate timing gates:

- `runtime.control_fps`: target environment step rate.
- `runtime.policy_fps`: policy inference rate; cached action is reused between
  policy ticks.
- `runtime.spacemouse_fps`: input-device polling rate.
- `runtime.image_obs_fps`: policy image render rate; cached images are reused
  when this is lower than control frequency.
- `runtime.render_fps`: GUI refresh rate for decoupled viewer paths.

For single-GPU interactive runs, reducing `IMAGE_OBS_FPS` and using separate
learner/inference devices can reduce render-time stutter:

```bash
IMAGE_OBS_FPS=10 LEARNER_DEVICE=cuda:0 INFERENCE_DEVICE=auto \
bash robosuite/pipeline/scripts/train_hil_serl.sh
```

## Multi-Thread Logic

The pipeline uses threads only around work that should not block the actor loop
unless explicitly flushed.

### Async Learner

Each trainer owns an optional learner thread:

- `algorithms/hil_serl/trainer.py`: `hil_serl_learner`
- `algorithms/flow_dagger/trainer.py`: `flow_dagger_learner`
- `algorithms/hg_dagger/trainer.py`: `hg_dagger_learner`
- `algorithms/awr/trainer.py`: `awr_learner`

Enable / disable it with:

```bash
ASYNC_UPDATES=true
ASYNC_UPDATES=false
```

The actor loop enqueues update requests. The learner thread consumes them under
a `threading.Condition`, runs gradient updates, and periodically publishes
fresh actor parameters. Queue sizes are capped in the trainer configs where
needed so a slow learner cannot accumulate unbounded stale work.

If you are debugging timing-sensitive crashes, first run:

```bash
ASYNC_UPDATES=false bash robosuite/pipeline/scripts/train_hil_serl.sh
```

### Inference / Model Locks

Online agents keep separate locks for parameter state and inference. The common
pattern is:

- `RLock` around mutable model state / actor snapshots.
- `Lock` around inference calls.

This allows the learner to publish updated weights without corrupting an
in-flight action selection.

### Async Checkpoint Writer

`AsyncCheckpointWriter` in `utils/train_utils.py` writes checkpoints from a
daemon thread so training does not block on disk I/O. The training loop queues
checkpoint requests, and close/flush paths wait for pending writes before exit.

### Async Transition Chunk Writer

`AsyncTransitionChunkWriter` in `utils/train_utils.py` buffers online/demo
transitions and writes chunk files under:

```text
<run_dir>/buffers/online_chunks/
<run_dir>/buffers/demo_chunks/
```

It uses a `threading.Condition`, pending deques, explicit flush requests, and
error propagation back to the main thread. This is the main persistence path for
long online runs.

### Async Viewer

`RobosuiteViewerRuntime` can render the GUI in a background thread only for the
`mjviewer` backend:

```bash
VIEWER_ASYNC=true VIEWER_BACKEND=mjviewer
```

The main loop publishes only the newest simulation snapshot. The viewer thread
does not consume every state; it renders the latest available state at
`runtime.render_fps`. This keeps GUI refresh bounded when the actor loop runs
faster than the display.

## Algorithm Summary

- HIL-SERL: online SAC-style learning with offline demos, optional human
  intervention, async updates, and WandB/stdout runtime logging.
- Flow-DAgger: flow-policy behavior cloning / DAgger-style online correction
  with optional frozen-policy evaluation mode.
- HG-DAgger: human-gated DAgger variant using the shared intervention and
  async training utilities.
- AWR: advantage-weighted regression with success/fail/demo data, optional
  Q/V cache building, and optional discriminator reward plumbing.

Algorithm details live under `algorithms/<name>/`; this README focuses on
runtime, rendering, and threading.

## Config Files

Hydra configs are in:

```text
robosuite/pipeline/config/
```

Common runtime fields to inspect first:

- `env.environment`
- `env.robots`
- `env.camera_names`
- `env.render_camera`
- `env.renderer`
- `runtime.interactive`
- `runtime.viewer_enabled`
- `runtime.viewer_backend`
- `runtime.viewer_async`
- `runtime.render_fps`
- `runtime.image_obs_fps`
- `runtime.async_updates`
- `intervention.enabled`

## Logging and Outputs

Runs write under the configured output root, typically `outputs/<algorithm>/`.
The training scripts also set useful debugging environment variables:

```bash
PYTHONFAULTHANDLER=1
HYDRA_FULL_ERROR=1
```

For AWR, the script defaults to offline WandB logging:

```bash
WANDB_MODE=offline
WANDB_ENTITY=songgao-personal
```

Other scripts expose `LOGGING_USE_WANDB` and pass Hydra logging flags into the
training module.
