# DIPOLE Pipeline

This directory implements the online human-in-the-loop training stack used by DIPOLE and DIPOLE-RL. Both algorithms train the same polarity-conditioned flow policy and follow the Flow-DAgger rollout protocol. DIPOLE uses a frozen nnPU failure discriminator to weight the two policy branches; DIPOLE-RL additionally learns offline-to-online IQL critics and uses their advantage in the branch weighting signal.

The pipeline is intended for research runs with robosuite, a SpaceMouse, and one or two CUDA devices. The default Python interpreter is:

```bash
/home/dodo/miniconda3/envs/dagger/bin/python
```

## Algorithms

| Mode | `algorithm.dipole.g_mode` | Training signal |
|---|---|---|
| DIPOLE | `nnpu_frozen` | Frozen nnPU failure score |
| DIPOLE-RL | `advantage` | IQL advantage minus frozen nnPU failure score |

The shared actor is the flow model in `robosuite.policy.flow_multi_update`. Its architecture and checkpoint parameter names are compatible with the pre-refactor flow-multi-update checkpoints.

```text
observations + proposed action chunk
             |
             +--> DIPOLE flow policy --> environment or human override
             |
             +--> frozen DynEncoder
                    |-- state feature ----------------------> V(s)
                    `-- action-conditioned chunk feature --> Q(s, a_chunk)
                                                        `--> frozen nnPU head

DIPOLE:    G = -failure_score
DIPOLE-RL: G = alpha * normalized_advantage
               - beta * normalized_failure_score
```

The nnPU encoder and head are always frozen. There is no online discriminator optimizer, discriminator replay buffer, or BCE update in either mode.

## Required checkpoints

Every run requires two independent artifacts:

1. `INIT_CHECKPOINT`: a compatible flow-dagger or flow-multi-update actor checkpoint. It supplies the base policy weights, camera order, action normalization, proprio normalization, and model configuration.
2. `NNPU_CKPT`: a task-calibrated `pu_bce_head.pth` produced by `robosuite/discriminator/dyn_disc/`.

The nnPU artifact must contain `pu_bce_detector`, `feature_source`, `transformer_layer`, and `model_ckpt`. The feature source and transformer layer are immutable properties of the trained head and are always read from this artifact. By default the embedded `model_ckpt` reconstructs `DynEncoder`; set `NNPU_ENCODER_CKPT` only when the embedded path is unavailable on the current machine, and only to the matching dynamics checkpoint.

The runtime threshold is read directly from:

```python
checkpoint["pu_bce_detector"]["thresholds"][task_name]
```

It is a task-specific success-calibration threshold. The pipeline does not read `meta.json`, does not use a Youden threshold, and does not recalibrate the threshold online.

## Quick start

Run commands from the repository root.

### Train DIPOLE

```bash
export INIT_CHECKPOINT=/abs/path/to/flow_checkpoint.pt
export NNPU_CKPT=/abs/path/to/pu_bce_head.pth
export ENVIRONMENT=PickPlaceCereal
export WANDB_MODE=offline

bash robosuite/pipeline/scripts/train_dipole.sh
```

Useful overrides include `LEARNER_DEVICE`, `INFERENCE_DEVICE`, `ACTION_HORIZON`, `EXECUTE_HORIZON`, `OMEGA`, `BETA`, `G_NORMALIZATION`, `DISC_INFERENCE_FPS`, and Hydra overrides passed after the script name.

### Initialize IQL Q/V offline

DIPOLE-RL should normally start from a task-specific Q/V warmup checkpoint:

```bash
export NNPU_CKPT=/abs/path/to/pu_bce_head.pth
bash robosuite/pipeline/scripts/utils/init_iql_qv.sh
```

The utility reads the configured `expert`, `success_rollout`, and `fail_rollout` HDF5 splits, freezes the dynamics encoder, optionally pre-encodes its state/chunk features, and writes:

```text
outputs/DIPOLE_rl/iql_qv_cache/<task>/iql_state.pt
```

The warmup is seed-controlled (`SEED`, default 42): it seeds python/numpy/torch so the demo-load order and the sampling stream are reproducible run-to-run on the same machine/GPU (`cudnn.benchmark`/TF32 stay on for speed, so this is not bit-exact across hardware).

With `SAVE_DATA=true` (default, via `warmup.num_trajectories.save_data`), the assembled raw offline transitions are also saved to `<data_root>/<task>/<SAVE_DIR>/iql_offline_transitions.pt` (plus a `.meta.json` sidecar). These reload verbatim via `FlowDaggerReplayBuffer.load` and feed the offline DIPOLE entrance under `pipeline/offline`.

This is an experiment-specific entrance: `ENVIRONMENT` and `INIT_CHECKPOINT` are intentionally hardcoded near the top of the script. Edit those constants directly for another task or base policy. The seed, trajectory counts, learner device, output path, warmup lengths, and offline-data save options retain their documented environment-variable overrides.

### Train DIPOLE-RL

```bash
export INIT_CHECKPOINT=/abs/path/to/flow_checkpoint.pt
export NNPU_CKPT=/abs/path/to/pu_bce_head.pth
export IQL_WARMUP_CKPT=/abs/path/to/iql_state.pt
export WANDB_MODE=offline

bash robosuite/pipeline/scripts/train_dipole_rl.sh
```

`G_MODE=advantage` is the RL path. `G_MODE=nnpu_frozen` keeps the same runtime and actor but disables advantage-based weighting.

### Evaluate a trained actor

```bash
CHECKPOINT=/abs/path/to/run/checkpoints/latest.pt \
ENVIRONMENT=PickPlaceCereal \
OMEGA_SWEEP="0,1,2,4" \
EPISODES=20 \
bash robosuite/pipeline/scripts/eval_dipole.sh
```

Evaluation is headless, can save MP4 rollouts, runs a branch-divergence diagnostic, and writes one `summary.json` per guidance value plus an optional `sweep_summary.json`.

### Inspect Q/V and discriminator reward

```bash
NNPU_CKPT=/abs/path/to/pu_bce_head.pth \
bash robosuite/pipeline/scripts/utils/vis_iql_qv.sh
```

This utility is diagnostic only. Its task, split, seed, and checkpoint layout are intentionally hardcoded near the top of the script; edit them directly for a different experiment. It must use the same task, camera mapping, nnPU artifact, and Q/V schema as training.

## Human-in-the-loop runtime

Interactive training follows the current Flow-DAgger behavior:

- Policy chunks are reset only on intervention start and intervention end.
- During intervention, the SpaceMouse action is the only action sent to the environment.
- The policy continues to plan an action chunk during intervention so the background nnPU scorer evaluates the policy proposal rather than the human action.
- With `DISC_INFERENCE_FPS=-1`, scoring is triggered once for each new policy chunk. A positive value requests fixed-rate scoring in Hz.
- The HUD reports the failure score, calibrated threshold, and failure state.
- Consecutive failure predictions can pause environment execution. Resume with ENTER or the configured SpaceMouse resume input. When safe re-arming is enabled, a safe prediction is required before the detector can trigger again.

The runtime requires a desktop `DISPLAY` when interactive input is enabled. Set `INTERACTIVE=false`, disable intervention, and use `MUJOCO_GL=egl` for headless runs.

HUD or optional display initialization failures are non-fatal: training may continue without visualization. A missing or incompatible nnPU checkpoint is fatal when it is required for G, Q/V features, or reward computation.

## Core behavior

### Frozen nnPU scoring

The dynamics encoder exposes two distinct feature domains:

- `encode_state_batch(images, proprio)` returns an action-free visual/proprio latent for V.
- `encode_chunk_batch(images, proprio, action_chunk)` returns the transformer latent conditioned on the complete action chunk for Q and nnPU.

For head logit `g(z)`, the public failure convention is:

```text
failure_score = -g(z)                 # larger means more failure-like
is_failure    = failure_score >= tau_task
r_disc        = -sigmoid(failure_score - tau_task)
```

The discriminator path is tensor-native; training does not round-trip features through NumPy.

### DIPOLE policy

The policy shares one flow backbone and adds a two-row polarity embedding. It trains positive and negative velocity branches and combines them at inference:

```text
v_guided = (1 + omega) * v_pos - omega * v_neg
```

Human-intervention samples explicitly force the positive branch weight to one. For non-intervention samples, the two flow-matching losses are weighted by:

```text
w_pos = sigmoid(clamp(beta_dipole * normalize(G) + k, -g_clip, g_clip))
w_neg = 1 - w_pos
L     = mean(w_pos * L_pos + w_neg * L_neg)
```

With the default `g_sign=negate_raw`, discriminator-only DIPOLE uses `G=-failure_score`. DIPOLE-RL uses `G=alpha*normalize(Q-V)-beta*normalize(failure_score)`. The advantage and failure channels have independent normalization state.

### DIPOLE-RL critics

Q consumes the action-conditioned chunk feature directly. V and target-V consume the action-free state feature. The frozen dynamics encoder owns action chunk normalization and fusion; Q does not add a second action projection. For an H-step replay window, the unchanged learner computes:

```text
R_H   = sum(i=0..H-1) gamma^i * r_total_i
y     = R_H + gamma^H * (1 - done_H) * target_V(s_next)
L_Q   = mean_k MSE(Q_k(s, a_chunk), y)
delta = min(Q_subset(s, a_chunk)) - V(s)
L_V   = mean(abs(expectile_tau - 1[delta < 0]) * delta^2)
```

Only target-V receives a Polyak update. Each learner tick performs the IQL update before the flow-policy update; there is no discriminator update.

## Configuration and repository layout

```text
pipeline/
|-- algorithms/
|   |-- dipole/           polarity flow policy, G providers, replay, trainer
|   |-- discriminator/    frozen nnPU adapter and shared DynEncoder
|   |-- q_learning/       IQL networks, replay, warmup, diagnostics
|   `-- flow_dagger/      shared rollout/runtime utilities
|-- config/
|   |-- train_dipole.yaml
|   `-- train_dipole_rl.yaml
|-- scripts/              maintained bash entry points
|-- train_dipole.py
|-- train_dipole_rl.py
`-- eval_dipole.py
```

Important configuration keys:

```yaml
algorithm:
  dipole:
    g_mode: nnpu_frozen       # nnpu_frozen | advantage
  discriminator:
    checkpoint: null          # NNPU_CKPT / pu_bce_head.pth
    encoder_ckpt: null        # optional path override
    task_name: ${env.environment}
    camera_to_view: {}
    inference:
      fps: -1
      intervene_env: true
  q_learning:
    enabled: true
    warmup_ckpt: null
```

`camera_to_view` maps a policy camera name to the view name expected by the dynamics encoder. Leave it empty when names match.

## Outputs and logging

Training runs write under `logging.output_root` and normally contain:

```text
<run>/
|-- checkpoints/
|-- tensorboard/
|-- buffers/
|-- run_info.json
`-- logs and evaluation artifacts
```

TensorBoard is enabled by the training scripts unless overridden. To use WandB, set `LOGGING_USE_WANDB=true`; keep `WANDB_MODE=offline`. The configured entity is `songgao-personal`. Offline runs can be synchronized manually after training.

## Checkpoint compatibility

- Existing flow policy checkpoints remain compatible because the actor architecture and state-dict keys are unchanged; only the import package is now `robosuite.policy.flow_multi_update`.
- Old LPB v2 BCE or online-BCE discriminator artifacts are not valid `NNPU_CKPT` inputs.
- Old Q/V warmup checkpoints that contain one LPB context feature, separate `q1/q2`, or omit `state_feature_dim` and `chunk_feature_dim` are rejected. Re-run `scripts/utils/init_iql_qv.sh` with the nnPU encoder.
- A Q/V checkpoint is tied to the nnPU encoder, task, camera views, action dimension, horizon, and ensemble size recorded in its metadata.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| `NNPU_CKPT must point to ... pu_bce_head.pth` | Provide an existing task-calibrated nnPU artifact. |
| Task threshold missing | Use the task name stored in `pu_bce_detector.thresholds`, or recalibrate the nnPU artifact for the task. |
| Dynamics checkpoint cannot be found | Set `NNPU_ENCODER_CKPT` to the matching encoder checkpoint; do not substitute a different model. |
| Missing encoder view | Add the camera to `env.camera_names` or configure `algorithm.discriminator.camera_to_view`. |
| Legacy Q/V schema error | Re-run offline Q/V warmup; old LPB/single-feature caches are intentionally incompatible. |
| Interactive run has no input/window | Verify `DISPLAY`, use `MUJOCO_GL=glfw`, and confirm SpaceMouse permissions. |
| CUDA device mismatch | Align learner, inference, encoder, and Q/V device settings; remember that `CUDA_VISIBLE_DEVICES` renumbers devices. |
| HUD fails but learning continues | This is expected for optional visualization. Check the log for the display error separately. |
| NaN/Inf critic values | Verify the actor and nnPU normalizers match their checkpoints, then inspect reward scale and the Q/V diagnostic outputs. |

# TODO List

1. [done] refactor to suit for new PU-learning based discriminator
2. [done] add [config](./config/discriminator.yaml) for online discriminator behavior setting, apply these settings in main training loop. Follow implementation and logic in git branch `dagger/v0-pu-bce`.
  - `algorithm.discriminator` is now a standalone Hydra fragment (`# @package algorithm.discriminator`) pulled into both `train_dipole.yaml` and `train_dipole_rl.yaml` via `defaults`, so the online scoring cadence / pause / HUD knobs live in one file. `feature_source` / `transformer_layer` stay read-only from the nnPU checkpoint (not exposed as overrides).
3. [done] modify [offline iql learning](./scripts/utils/init_iql_qv.sh) and related logic:
  - save IQL training data each time, new parameters have been defined in [config](./config/train_dipole_rl.yaml) (`warmup.num_trajectories.save_data` / `save_dir`); the assembled raw offline transitions are written to `<data_root>/<task>/<save_dir>/iql_offline_transitions.pt` (reloadable via `FlowDaggerReplayBuffer.load`) for offline DIPOLE reuse.
  - make all learning deterministic, controlled by `SEED`, reproducable (warmup seeds python/numpy/torch; demo-load order and sampling stream are reproducible run-to-run on the same machine/GPU).
  - debug and check (although no output bugs are observed)
4. [tbd] implement offline dipole learning under folder `./offline`
  - load pretrain_data + offline_data into `replay_buffer`, keep `demo_buffer` empty
  - use iql checkpoint and pretrained policy to perform dipole_rl algorithm 
5. [tbd] change current condition-injection based positive-negative policy architecture. Target: LoRA liked negative injection
  - for positive policy, uses original policy; during updates, tune all policy network
  - for negative policy, add LoRA(or others) module. During updates, tune base policy NNs + LoRA module together. note: when BP for negative policy, lr for base policy NNs should set 0.5 of LoRA module
  - during online rollout, uses only positive branch (no extra 2-policy inference during online)
  - during evaluation, uses DIPOLE 2 policy inference with $1+w$ and $w$ weighted