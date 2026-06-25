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
                    `-- action-conditioned chunk feature --> frozen nnPU head
                                                        `--> Q token projector + action projector

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
bash robosuite/pipeline/scripts/utils/init_iql_qv.sh
```

The utility reads the configured `expert`, `success_rollout`, and `fail_rollout` HDF5 splits, freezes the dynamics encoder, optionally pre-encodes its state/chunk features, and writes:

```text
outputs/dipole_rl/offline_iql_qv-v1/<task>/iql_state.pt
```

The warmup is seed-controlled (`SEED`, default 42): it seeds python/numpy/torch so the demo-load order and the sampling stream are reproducible run-to-run on the same machine/GPU (`cudnn.benchmark`/TF32 stay on for speed, so this is not bit-exact across hardware).

With `warmup.num_trajectories.save_data=true` (forced by `init_iql_qv.sh`), the assembled raw offline transitions are also saved to `<data_root>/<task>/<save_dir>/iql_offline_transitions.pt` (plus a `.meta.json` sidecar). These reload verbatim via `FlowDaggerReplayBuffer.load`. The export is implemented, but the offline DIPOLE training entry under `pipeline/offline` is still a placeholder.

With `FREEZE_POST_SUCCESS=true` (default, via `warmup.freeze_post_success`), each success demo's post-success tail is collapsed into a single frozen absorbing anchor before the IQL critics consume it. Recorded `success_rollout` demos are fixed-length (e.g. 400 steps) but keep running the live policy after success, so the frames after the first success are real *drift* (moving state, non-zero actions). Fitting Q/V on those many distinct, meaningless states is a burden, while their dense sampling is what anchors the terminal value `V≈0` and stabilizes TD. The warmup therefore overwrites every frame strictly after the first success frame `t_s` with a copy of that frame: `obs` (images + proprio), `next_obs`, and `action` all become the `t_s` values, and the frozen-tail reward is set to `0.0`. The first success frame keeps its real `(s, a)`; `fail_rollout` and demos without a success frame are untouched. The freeze is applied after the offline-data save, so the persisted `offline_data` keeps the raw drift frames; only the in-memory buffer the IQL warmup trains on is frozen. Set `FREEZE_POST_SUCCESS=false` to disable.

Current IQL warmup uses schema version 3. It compresses the frozen encoder features with Token/Group projectors before the Q/V heads, re-injects the raw `action_chunk` into Q through an action projector, and defaults to a lighter head (`hidden_dims=[256,256]`, `chunk_proj_dim=128`, `state_proj_dim=128`). The Hydra defaults run `10000` V-only steps followed by `5000` full IQL steps, with `reward_mode="-1/0"`, `output_reward_coef=0.1`, and `disc_reward_coef=0.0`.

This is an experiment-specific entrance: `ENVIRONMENT`, `INIT_CHECKPOINT`, and `NNPU_CKPT` are intentionally hardcoded near the top of the script. Edit those constants directly for another task, base policy, or nnPU artifact. `SEED`, `BATCH_SIZE`, `DEVICE`, `OUTPUT_DIR`, `OUTPUT_FILE`, `PREENCODE_CACHE_DEVICE`, and `FREEZE_POST_SUCCESS` retain environment-variable overrides.

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

This utility is diagnostic only. Its task, split, seed, checkpoint layout, and nnPU path are intentionally hardcoded near the top of the script; edit them directly for a different experiment. It must use the same task, camera mapping, nnPU artifact, and Q/V schema as training.

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

Q first compresses the action-conditioned chunk feature with a shared Token/Group projector, separately embeds the raw action chunk with an `ActionProjector`, then feeds the concatenated projection to each Q head. V and target-V consume the action-free state feature through their own Token/Group projector. For an H-step replay window, the learner computes:

```text
R_H   = sum(i=0..H-1) gamma^i * r_total_i
y     = R_H + gamma^H * (1 - done_H) * target_V(s_next)
L_Q   = mean_k MSE(Q_k(s, a_chunk), y)
delta = min(Q_subset(s, a_chunk)) - V(s)
L_V   = mean(abs(expectile_tau - 1[delta < 0]) * delta^2)
```

Only target-V receives a Polyak update. Each learner tick performs the IQL update before the flow-policy update; there is no discriminator update. Q uses an ensemble (`q_ensemble_size=10` by default), V uses the minimum over a random subset (`v_subset_size=2` by default), and the actor advantage uses the ensemble output through `compute_ensemble_advantage`.

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
|   |-- train_dipole_rl.yaml
|   |-- discriminator.yaml
|   `-- offline.yaml
|-- offline/              placeholder offline DIPOLE entry points
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
    config:
      reward_mode: "-1/0"
      hidden_dims: [256, 256]
      chunk_proj_dim: 128
      state_proj_dim: 128
      action_proj_dim: 64
      q_ensemble_size: 10
      v_subset_size: 2
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

- Existing flow policy checkpoints remain compatible because the actor architecture and state-dict keys are unchanged.
- Old LPB v2 BCE or online-BCE discriminator artifacts are not valid `NNPU_CKPT` inputs.
- Old Q/V warmup checkpoints that contain one LPB context feature, separate `q1/q2`, omit `state_feature_dim` / `chunk_feature_dim`, or predate schema version 3 are rejected. Re-run `scripts/utils/init_iql_qv.sh` with the nnPU encoder.
- A Q/V checkpoint is tied to the nnPU encoder, task, camera views, action dimension, horizon, ensemble size, token count, proprio width, and projector dimensions recorded in its metadata.

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

- [x] step1: Refactor the pipeline to use the PU-learning based nnPU discriminator.
- [x] step2: Add shared online discriminator configuration in [config/discriminator.yaml](./config/discriminator.yaml). `algorithm.discriminator` is a standalone Hydra fragment merged into both `train_dipole.yaml` and `train_dipole_rl.yaml`; online scoring cadence, pause behavior, and HUD settings now live in one file. `feature_source` and `transformer_layer` remain read-only properties loaded from the nnPU checkpoint.
- [x] step 3: Update offline IQL warmup in [scripts/utils/init_iql_qv.sh](./scripts/utils/init_iql_qv.sh):
  1. Save assembled raw offline transitions to `<data_root>/<task>/<save_dir>/iql_offline_transitions.pt` plus metadata for later reuse.
  2. Seed python, numpy, and torch from `SEED` so demo loading and sampling are reproducible on the same machine/GPU.
  3. Freeze post-success drift in memory with `warmup.freeze_post_success`, while keeping saved offline data raw.
  4. Move IQL Q/V to schema version 3 with Token/Group feature projectors, raw action re-injection for Q, smaller `[256,256]` heads, `q_ensemble_size=10`, and `v_subset_size=2`.
  5. Remove legacy debug-only Q/V scripts and keep `vis_iql_qv.sh` as the maintained diagnostic entrance.
- [x] step 4: Implement offline DIPOLE training under `pipeline/offline`, with config being [this](./config/offline.yaml)
  1. Load `pretrain_data` and exported `offline_data` into `replay_buffer`, keeping `demo_buffer` empty. **note**: for success trajectories, only uses "is_success=false" part; for fail trajectories, only uses "failure_frame_mask=0"(no failure) part
  2. Use the pretrained flow policy and IQL checkpoint to run the DIPOLE-RL update offline. **note**: when compute dipole weight G, use TD term $Adv= V - \gamma*V' - r$ rather then current $Adv=Q-V$
  3. for code structure, make codebase moduler, leave tool functions under `robosuite/pipeline/offline/utils`
  4. implement evaluation logic: evaluate success rate for finetuned policy. For video saving logic and saving directory arrangement, you can refer to `pipeline/sft` folder in branch `dagger/v0-pu-bce`
- [ ] step 5: Replace the current condition-injection positive/negative policy architecture with a LoRA-like negative branch.
  - Positive branch should use the original policy path and update all policy network parameters.
  - Negative branch should add an adapter module; during negative updates, train the adapter and base policy together, with the base-policy learning rate lower than the adapter learning rate.
  - Online rollout should use only the positive branch.
  - Evaluation should keep DIPOLE two-branch inference with `(1 + omega)` and `omega` weighting.
