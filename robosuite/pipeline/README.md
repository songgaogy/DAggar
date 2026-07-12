# DIPOLE Pipeline

This directory implements the online human-in-the-loop training stack used by DIPOLE and DIPOLE-RL. Both algorithms train the same polarity-conditioned flow policy and follow the Flow-DAgger rollout protocol. DIPOLE uses a frozen nnPU failure discriminator to weight the two policy branches; DIPOLE-RL additionally learns an offline-to-online **V-only** IQL value (no Q head) and uses its **TD-residual advantage** in the branch weighting signal. (Under the deterministic dynamics + single-action coverage here the Q step is redundant — `Q(s,a)=r+γV(s')` holds exactly, so Q cancels out of the value fit; see [V_ONLY_ADVANTAGE_DESIGN.md](./V_ONLY_ADVANTAGE_DESIGN.md).)

The pipeline is intended for research runs with robosuite, a SpaceMouse, and one or two CUDA devices. The default Python interpreter is:

```bash
/home/dodo/miniconda3/envs/dagger/bin/python
```

## Algorithms

| Mode | `algorithm.dipole.g_mode` | Training signal |
|---|---|---|
| DIPOLE | `nnpu_frozen` | Frozen nnPU failure score |
| DIPOLE-RL | `advantage` | V-only TD-residual advantage minus frozen nnPU failure score |

The shared actor is the flow model in `robosuite.policy.flow_multi_update`. Its architecture and checkpoint parameter names are compatible with the pre-refactor flow-multi-update checkpoints.

```text
observations + proposed action chunk
             |
             +--> DIPOLE flow policy --> environment or human override
             |
             +--> frozen DynEncoder
                    |-- state feature ----------------------> V(s), target_V(s')
                    `-- action-conditioned chunk feature --> frozen nnPU head

DIPOLE:    G = -failure_score
DIPOLE-RL: G = alpha * normalized_advantage        # advantage = r + γ^H·V(s') - V(s)
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

Useful overrides include `LEARNER_DEVICE`, `INFERENCE_DEVICE`, `ACTION_HORIZON`, `EXECUTE_HORIZON`, `OMEGA`, `BETA`, `DISC_INFERENCE_FPS`, and Hydra overrides passed after the script name.

### Initialize IQL V offline

DIPOLE-RL should normally start from a task-specific V warmup checkpoint:

```bash
bash robosuite/pipeline/scripts/utils/init_iql_qv.sh
```

The utility reads the configured `expert`, `success_rollout`, and `fail_rollout` HDF5 splits, freezes the dynamics encoder, optionally pre-encodes its state/chunk features, and writes:

```text
outputs/dipole_rl/offline_iql_qv-v1/<task>/iql_state.pt
```

The warmup is seed-controlled (`SEED`, default 42): it seeds python/numpy/torch so the demo-load order and the sampling stream are reproducible run-to-run on the same machine/GPU (`cudnn.benchmark`/TF32 stay on for speed, so this is not bit-exact across hardware).

With `warmup.num_trajectories.save_data=true` (forced by `init_iql_qv.sh`), the assembled raw offline transitions are also saved to `<data_root>/<task>/<save_dir>/iql_offline_transitions.pt` (plus a `.meta.json` sidecar). These reload verbatim via `FlowDaggerReplayBuffer.load`. The export is implemented, but the offline DIPOLE training entry under `pipeline/offline` is still a placeholder.

With `FREEZE_POST_SUCCESS=true` (default, via `warmup.freeze_post_success`), each success demo's post-success tail is collapsed into a single frozen absorbing anchor before the IQL value consumes it. Recorded `success_rollout` demos are fixed-length (e.g. 400 steps) but keep running the live policy after success, so the frames after the first success are real *drift* (moving state, non-zero actions). Fitting V on those many distinct, meaningless states is a burden, while their dense sampling is what anchors the terminal value `V≈0` and stabilizes TD. The warmup therefore overwrites every frame strictly after the first success frame `t_s` with a copy of that frame: `obs` (images + proprio), `next_obs`, and `action` all become the `t_s` values, and the frozen-tail reward is set to `0.0`. The first success frame keeps its real `(s, a)`; `fail_rollout` and demos without a success frame are untouched. The freeze is applied after the offline-data save, so the persisted `offline_data` keeps the raw drift frames; only the in-memory buffer the IQL warmup trains on is frozen. Set `FREEZE_POST_SUCCESS=false` to disable.

Current IQL warmup uses schema version 5 (V-only; no Q head; N-head V-ensemble with expectile-TD + soft-LCB). It compresses the frozen encoder state feature with a Token/Group projector before each V head and defaults to a lighter head (`hidden_dims=[256,256]`, `state_proj_dim=128`). The value is learned V-only by a single expectile-TD loop over `warmup_value_steps + warmup_full_steps` steps (the former two-phase value-only → full-IQL split no longer applies); see *DIPOLE-RL value* below for the expectile/ensemble/LCB knobs. Reward composition (`reward_mode`, `output_reward_coef`, `disc_reward_coef`) is set in `train_dipole_rl.yaml`.

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

### Inspect V and discriminator reward

```bash
NNPU_CKPT=/abs/path/to/pu_bce_head.pth \
bash robosuite/pipeline/scripts/utils/vis_iql_qv.sh
```

This utility is diagnostic only. Its task, split, seed, checkpoint layout, and nnPU path are intentionally hardcoded near the top of the script; edit them directly for a different experiment. It must use the same task, camera mapping, nnPU artifact, and V (schema v5) checkpoint as training. It produces a 4-subplot figure: `V_lcb` (with the ensemble ±std band) / target-V, the V differences, the **advantage subplot** (the one-macro-step TD residual overlaid with the GAE(λ) advantage on `V_lcb`), and the per-frame discriminator reward (no Q curves; the old best-of-n Q diagnostic has been removed). GAE recurs in chunk periods (`γ_eff = γ^action_horizon`) and is evaluated densely at every window start; set `GAE_LAMBDA` (default `0.95`) to change λ.

## Human-in-the-loop runtime

Interactive training follows the current Flow-DAgger behavior:

- Policy chunks are reset only on intervention start and intervention end.
- During intervention, the SpaceMouse action is the only action sent to the environment.
- The policy continues to plan an action chunk during intervention so the background nnPU scorer evaluates the policy proposal rather than the human action.
- With `DISC_INFERENCE_FPS=-1`, scoring is triggered once for each new policy chunk. A positive value requests fixed-rate scoring in Hz.
- The HUD reports the failure score, calibrated threshold, and failure state.
- Consecutive failure predictions can pause environment execution. Resume with ENTER or the configured SpaceMouse resume input. When safe re-arming is enabled, a safe prediction is required before the detector can trigger again.

The runtime requires a desktop `DISPLAY` when interactive input is enabled. Set `INTERACTIVE=false`, disable intervention, and use `MUJOCO_GL=egl` for headless runs.

HUD or optional display initialization failures are non-fatal: training may continue without visualization. A missing or incompatible nnPU checkpoint is fatal when it is required for G, V features, or reward computation.

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

The policy is **two independent, fully finetuned flow policies** — a positive policy and a negative policy:

```text
positive policy = MultiModalFlowPolicy (full-tune)
negative policy = MultiModalFlowPolicy (full-tune)
```

Each is a plain `MultiModalFlowPolicy` built by `build_flow_policy`, trained full-tune under the **base flow-policy freeze regime**: the ResNet image encoder trains only `layer3`/`layer4` (stem/`layer1`/`layer2` frozen), the CLIP text encoder is fully frozen, and everything else (proprio tokenizer, fusion, condition aggregator, flow head) trains. There are **no LoRA adapters** — each policy has its own AdamW optimizer at the flow `learning_rate`/`weight_decay`, its own GradScaler, and its own double-buffered inference copy. Both policies start from the same pretrained init checkpoint, so `v_pos == v_neg` at step 0 and diverge as training proceeds.

- **Online rollout** uses the positive policy only, a single forward per ODE step (no omega combination); the negative policy is never encoded or forwarded when `omega == 0`.
- **Evaluation** combines the two policies with CFG-style guidance at inference:

```text
v_guided = (1 + omega) * v_pos - omega * v_neg
```

`omega` is therefore an eval-only knob (`algorithm.dipole.guidance_omega`). Human-intervention samples explicitly force the positive branch weight to one. For non-intervention samples, the two flow-matching losses are weighted by:

```text
w_pos = sigmoid(clamp(beta_dipole * normalize(G) + k, -g_clip, g_clip))
w_neg = 1 - w_pos
L     = w_pos * L_pos + w_neg * L_neg     # per-policy weighted mean
```

The positive policy trains on the `w_pos`-weighted loss, the negative policy on the `w_neg`-weighted loss — two independent forward/backward passes and two optimizer steps per update. The soft-weight scheme above (`offline.mode == "normal"`) and the hard-split schemes (`naive`/`neg_all`, where `w_pos`/`w_neg` become ~{0,1} membership) both reduce to this same two-policy weighted update. The G provider returns the preference `G` directly (larger `G` -> more positive branch): discriminator-only DIPOLE uses `G=-failure_score`, and DIPOLE-RL uses `G=alpha*normalize(A)-beta*normalize(failure_score)` where the advantage is the V-only TD residual `A = r + γ^H·target_V(s') - V(s)`. The advantage and failure channels have independent normalization state.

> Note: this two-policy scheme is a clean break from the earlier dual-LoRA-on-frozen-backbone design — old dual-LoRA checkpoints do not load. Training two full policies is ~2x the compute/VRAM of the LoRA scheme; online rollout stays positive-only so the negative policy's inference copies sit idle during rollout.

### DIPOLE-RL value (V-only, expectile + soft-LCB ensemble)

There is no Q head. The value is a **V-ensemble** of `v_ensemble_size` (default 2) independent heads, each with its own Token/Group projector; a full target-ensemble Polyak-tracks it. The scalar value consumed everywhere is the soft lower-confidence bound `V_lcb = mean_k V_k - beta*std_k V_k` (`ensemble_lcb_beta`, default 0.5 — soft, not hard min, so the recoverable-state lift in the high-disagreement region is kept). For an H-step replay window, every head regresses onto the shared **1-step** LCB bootstrap target by an **expectile** loss (optimism knob `expectile_tau`, default 0.85 > 0.5, turning the behavior value `V^beta` into the optimistic `V*`):

```text
R_H   = sum(i=0..H-1) gamma^i * r_total_i
y     = R_H + gamma^H * (1 - done_H) * V_lcb_target(s_next)     # shared across heads
mask_k ~ Bernoulli(ensemble_bootstrap_prob)                    # per-head, per-sample
L_V   = expectile_tau( y - V_k(s) )  weighted by mask_k         # summed over heads
```

The per-head Bernoulli **bootstrap mask** (keep-prob `ensemble_bootstrap_prob`, default 0.5) diversifies the heads so the LCB std does not collapse. `v_ensemble_size=1` recovers a single head (std ≡ 0, `beta`/mask inert). The TD target stays **1-step** (n-step propagation was tried and reverted — see [V_ONLY_ADVANTAGE_DESIGN.md](./V_ONLY_ADVANTAGE_DESIGN.md) §4); `λ` survives only in the read-out GAE.

> **Empirical update (`V_ONLY_ADVANTAGE_DESIGN.md` §4/§7-6):** the expectile *optimism* was tested and **rejected** — it gave no gain and degraded results. Runs therefore set `expectile_tau=0.5`, at which `expectile_tau(y - V_k)` reduces to `0.5·MSE` (asymmetry inert) → the value fit is plain **MSE-TD**. The **ensemble soft-LCB is the axis that works**, run with `v_ensemble_size=5`. Both are applied via `init_iql_qv.sh` env overrides (`EXPECTILE_TAU=0.5`, `V_ENSEMBLE_SIZE=5`); the in-code defaults (0.85 / 2) still hold the original hypothesis.

Only target heads receive a Polyak update. Each learner tick performs the V update before the flow-policy update; there is no discriminator update. The per-step advantage consumed by the flow branch weighting is the TD residual on the LCB value `A = r + gamma^H·(1-done)·V_lcb_target(s') - V_lcb(s)` (`IQLLearner.compute_td_advantage`); the offline path precomputes the same residual by window start index (`offline/utils/advantage.py`).

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
      state_proj_dim: 128
      proprio_proj_dim: 32
      expectile_tau: 0.85         # optimism knob (expectile-TD value fit)
      v_ensemble_size: 2          # N V heads for the soft-LCB
      ensemble_lcb_beta: 0.5      # V_lcb = mean_k - beta*std_k
      ensemble_bootstrap_prob: 0.5  # per-head bootstrap-mask keep-prob
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
- Any warmup checkpoint that contains a Q head (`q_ensemble`/`q1`) — i.e. every pre-V-only schema (v2/v3) — is rejected by the V-only learner. Re-run `scripts/utils/init_iql_qv.sh` with the nnPU encoder to produce a schema-v4 V-only checkpoint.
- A V checkpoint is tied to the nnPU encoder, task, camera views, horizon, token count, proprio width, and V-projector dimensions recorded in its metadata.

## Troubleshooting

| Symptom | Resolution |
|---|---|
| `NNPU_CKPT must point to ... pu_bce_head.pth` | Provide an existing task-calibrated nnPU artifact. |
| Task threshold missing | Use the task name stored in `pu_bce_detector.thresholds`, or recalibrate the nnPU artifact for the task. |
| Dynamics checkpoint cannot be found | Set `NNPU_ENCODER_CKPT` to the matching encoder checkpoint; do not substitute a different model. |
| Missing encoder view | Add the camera to `env.camera_names` or configure `algorithm.discriminator.camera_to_view`. |
| Q-head checkpoint rejected / `v_ensemble_size` mismatch | Re-run offline V warmup (schema v5); pre-V-only Q-containing caches and single-head v4 checkpoints are intentionally incompatible. |
| Interactive run has no input/window | Verify `DISPLAY`, use `MUJOCO_GL=glfw`, and confirm SpaceMouse permissions. |
| CUDA device mismatch | Align learner, inference, encoder, and V device settings; remember that `CUDA_VISIBLE_DEVICES` renumbers devices. |
| HUD fails but learning continues | This is expected for optional visualization. Check the log for the display error separately. |
| NaN/Inf V values | Verify the actor and nnPU normalizers match their checkpoints, then inspect reward scale and the V diagnostic outputs. |

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
- [x] step 5: Represent polarity as **two independent, fully finetuned flow policies** (positive + negative). **note**: this supersedes the earlier dual-LoRA-on-frozen-backbone scheme (which itself superseded the original single-negative-adapter / condition-injection plans) — it is a clean break, and old dual-LoRA checkpoints do not load.
  - Each policy is a plain `MultiModalFlowPolicy` (`build_flow_policy`), full-tune under the base freeze regime (ResNet `layer3`/`layer4` + heads trainable, ResNet stem/`layer1`/`layer2` + CLIP frozen). Both start from the same pretrained init, so `v_pos == v_neg` at step 0.
  - Positive policy trains on the `w_pos`-weighted loss, negative on `w_neg` — two independent forward/backward passes, two optimizers, at the flow `learning_rate`/`weight_decay`. Soft-weight (`normal`) and hard-split (`naive`/`neg_all`) both reduce to this.
  - Online rollout uses only the positive policy; `guidance_omega` is eval-only.
  - Evaluation keeps DIPOLE two-policy inference with `(1 + omega)` and `omega` weighting.
- [x] step 6: Remove the Q head across the codebase and learn the value V-only (schema v4). Under deterministic dynamics + single-action coverage, `Q(s,a)=r+γV(s')` holds exactly so Q is redundant; see [V_ONLY_ADVANTAGE_DESIGN.md](./V_ONLY_ADVANTAGE_DESIGN.md).
  1. `IQLLearner` drops the Q ensemble / action projector / optimizer; `update` becomes a single V-only MSE-TD step; add `compute_td_advantage` (`A = r + γ^H·(1-done)·target_V(s') - V(s)`). Warmup runs one V-only loop; checkpoint schema bumped to v4 and old Q checkpoints are rejected.
  2. `vis_iql_qv.sh` / `vis_qv.py` drop the Q curves and the best-of-n Q diagnostic; only V / target-V / TD-advantage / reward are plotted.
  3. The online `AdvantageGProvider` is a stub (`NotImplementedError`): the online `DipoleBatch` carries no next-state/reward, so threading those through the online sampler is deferred. The offline path (`OfflineAdvantageGProvider`) already serves the precomputed TD residual.
  4. `expectile_tau` is retained as a reserved knob for a future expectile-TD ("optimism") value fit; the current path is plain MSE-TD.
