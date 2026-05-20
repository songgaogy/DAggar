# DIPOLE — Dichotomous Diffusion Policy Optimization

DIPOLE trains a flow-matching policy with two CFG-style branches whose per-sample losses are weighted by a bounded sigmoid derived from a frozen LPB v2 BCE detector score `G`. At inference, the branches' velocity predictions are linearly combined to controllably scale policy aggressiveness without retraining.

```
w_pos = sigmoid(beta * G + k)               # positive branch weight
w_neg = 1 - w_pos                           # negative branch weight
loss_pos = w_pos * ||eps - eps_pos(a_t, s, t)||^2
loss_neg = w_neg * ||eps - eps_neg(a_t, s, t)||^2
eps_guided = (1 + omega) * eps_pos - omega * eps_neg
```

Convention: `G` larger = better state/action. BCE raw scores are "larger = more failure-like", so `G = -raw_bce_score` (configurable via `g_sign`). Intervention transitions force `w_pos=1, w_neg=0` via an explicit mask, not via `G = +inf` (avoids NaN under autocast).

---

## Design decisions

1. **Adapter architecture**: single shared backbone (`MultiModalFlowPolicy`) + one `nn.Embedding(2, cond_dim)` polarity embedding (row 0 = neg, row 1 = pos). The embedding is added to the `task_scene_cond` vector before the `flow_head`. Total branch-specific parameters: `2 × cond_dim ≈ 512 floats`. Backbone is trainable (gradients from both branches flow into the shared parameters).

2. **Detector source**: a **pre-fitted BCE checkpoint** produced by `robosuite/discriminator/lpb_v2/scripts/run_bce_robosuite_benchmark.sh` (or its `visualize_bce_robosuite.sh` twin). DIPOLE does not fit anything itself. The default path is `checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth`.

3. **Training data**: demo HDF5 + online rollouts + optional human intervention.

4. **Base policy**: loaded from a flow-dagger / flow-multi `.pt` checkpoint via `runtime.init_checkpoint`. **Mandatory** unless resuming a DIPOLE checkpoint.

---

## Files

```
robosuite/pipeline/
├── algorithms/dipole/
│   ├── __init__.py                # DipoleAgent / DipoleTrainer / LPBV2GProvider exports
│   ├── common.py                  # DipoleConfig, DipoleBatch, LPBDetectorConfig
│   ├── replay_buffer.py           # DipoleReplayBuffer with dual-track image preprocessing
│   ├── models/
│   │   └── flow.py                # DipolePolarityFlowModel + DipoleFlowPolicy
│   ├── g_provider.py              # LPBV2GProvider: loads bce_head.pth, runs encoder + BCE head
│   ├── agent.py                   # DipoleAgent
│   └── trainer.py                 # DipoleTrainer
├── train_dipole.py                # Hydra training driver
├── eval_dipole.py                 # Headless checkpoint eval driver
├── config/train_dipole.yaml       # Hydra config
└── scripts/
    ├── train_dipole.sh            # Training bash entry
    └── eval_dipole.sh             # Eval bash entry
```

The legacy `tools/fit_dipole_lpb_artifact.py / .sh` and `tools/dipole_g_smoke.py` were removed — DIPOLE now only consumes the BCE artefacts produced by the upstream `lpb_v2` benchmark runner.

---

## Polarity-conditioned policy

`DipolePolarityFlowModel` wraps `MultiModalFlowPolicy` and adds:

```python
self.polarity_embedding = nn.Embedding(2, cond_dim)

def forward_from_context(self, x_t, t, context, polarity_idx):
    cond = context["task_scene_cond"] + self.polarity_embedding.weight[polarity_idx]
    return self.flow_head(x_t=x_t, timesteps=t,
                          task_scene_cond=cond,
                          context_tokens=context["context_tokens"],
                          context_padding_mask=context["context_padding_mask"])
```

Initialization schemes (`cfg.algorithm.dipole.polarity_embedding_init`):

| Scheme           | Row 0 (neg)         | Row 1 (pos)         | Use case                                |
|------------------|---------------------|---------------------|-----------------------------------------|
| `small_gaussian` | `N(0, scale)`       | `N(0, scale)`       | Default; symmetric initial branch drift |
| `zero_neg`       | `0`                 | `N(0, scale)`       | Neg branch == base policy at init       |
| `antipodal`      | `-e, e~N(0, scale)` | `+e`                | Maximally separated init                |

Loading the base flow-policy checkpoint uses `strict=False` so the new `polarity_embedding.*` keys retain their fresh init.

> **CFG-at-start caveat**: with `polarity_embedding_init=small_gaussian` and `guidance_omega>0`, the CFG combine `(1+omega)*v_pos - omega*v_neg` *amplifies* the small init-time difference between the two branches by a factor of `(1+2*omega)`. Loss-side training is fine (`w_pos + w_neg = 1` per sample), but the very first rollouts may show extra jitter. If that becomes a problem, the lever is `polarity_embedding_init=zero_neg` (or a future zero-both + omega warmup); we're leaving the defaults as-is until we see the issue empirically.

---

## Training step

`DipoleFlowPolicy.update(batch: DipoleBatch)` runs one optimization step:

```text
1. Sample noise + timesteps; build x_t = (1-t)*noise + t*action.
2. Encode multimodal context ONCE.
3. forward_from_context twice (polarity 0/neg, 1/pos) -> v_pos, v_neg.
4. (no grad) LPBV2GProvider.compute_g_for_batch -> raw scores
   -> sign flip -> normalize -> clamp -> sigmoid -> (w_pos, w_neg).
   Intervention mask overrides: w_pos=1, w_neg=0.
5. Per-sample flow / endpoint / smooth MSE for each branch.
6. Weighted means via _weighted_mean.
7. loss = loss_pos + loss_neg, single .backward(), single AdamW optimizer.
```

**G normalization** (`cfg.algorithm.dipole.g_normalization`):

| Mode             | Formula                                | Notes                                 |
|------------------|----------------------------------------|---------------------------------------|
| `batch_zscore`   | `(G - G.mean()) / (G.std() + 1e-6)`    | Default; auto-scales different kinds  |
| `running_zscore` | EMA(mean,var) on the no-grad path      | Stable when intervention skews batches|
| `minmax`         | `2 * (G - lo) / (hi - lo) - 1`         | Symmetric clip to [-1, 1]             |
| `none`           | `G`                                    | Pass-through                          |

**Pre-sigmoid clamp**: `clamp(beta * G + k, -g_clip, +g_clip)` with default `g_clip=10.0`.

---

## Inference

`DipoleFlowPolicy.select_action(obs)` does CFG-guided ODE sampling (`_sample_guided_action_sequence`):

```text
encode context once.
x = noise (or zeros if deterministic).
for step in [0, n_ode_steps):
    t = step / n_ode_steps
    v_pos = model.forward_from_context(x, t, context, polarity_idx=1)
    v_neg = model.forward_from_context(x, t, context, polarity_idx=0)
    v = (1 + omega) * v_pos - omega * v_neg
    x = x + (1 / n_ode_steps) * v
return x.transpose(1, 2)
```

`omega` is read from `cfg.algorithm.dipole.guidance_omega` at each call; setting `OMEGA_RUNTIME_OVERRIDE` in the bash script sweeps aggressiveness without retraining.

---

## LPB v2 BCE G provider

`LPBV2GProvider` (`g_provider.py`):

```python
LPBV2GProvider(ckpt_path, task_name, device, camera_to_view={})
provider.bind_policy_cameras(camera_names_in_obs_order)
provider.compute_g_for_batch(batch) -> torch.Tensor (B,)   # larger raw = more failure-like
```

### Expected BCE checkpoint schema

The `ckpt_path` artefact is the file written by
`robosuite/discriminator/lpb_v2/scripts/run_bce_robosuite_benchmark.sh`
(or its `visualize_bce_robosuite.sh` companion). Top-level keys:

```python
{
    "epoch":             int,
    "in_dim":            int,
    "hidden":            int,
    "num_layers":        int,
    "bce_detector":      dict,        # BCEDiscriminator.state_dict()
    "feature_source":    str,         # "encoder" | "transformer"
    "transformer_layer": int,
    "model_ckpt":        str,         # path to LPB v2 dynamics .pth (rebuilds the encoder)
    "calib_mode":        str,
    "fail_bank_video_ids": list,
    "fail_calib_video_ids": list,
}
```

`bce_detector["thresholds"]` is a `Dict[task_name, float]` populated during the upstream benchmark run; DIPOLE looks it up by `algorithm.task_name` and raises if the task is missing.

### View-name mapping

The LPB encoder embedded in the checkpoint has its own `view_names` list (e.g. the shared `train-20260428_210501` ckpt was trained with `view_names=["agentview"]` only). The policy may use a different (and possibly larger) camera tuple, e.g. `["agentview", "robot0_robotview", "robot0_eye_in_hand"]`.

`bind_policy_cameras(policy_camera_names)` resolves the encoder-side view → policy-camera index map. The default is identity (encoder view name must exist verbatim in the policy camera tuple). If they differ, set:

```yaml
algorithm:
  dipole:
    lpb_detector:
      camera_to_view: { robot0_robotview: agentview }   # policy_camera -> encoder_view
```

The provider raises a clear error if a required encoder view is not represented in the policy camera tuple (either directly or via `camera_to_view`).

### Image-domain bridging

`DipoleBatch.image_obs_raw` is `[0, 1]` floats at the policy's `image_size`. The provider resizes per-view tensors to the encoder's `original_img_size` via bilinear interpolation before calling `encoder.encode_batch`. The encoder applies its own per-view normalization internally.

---

## Config schema (`config/train_dipole.yaml`)

The DIPOLE config keeps the shared flow-policy fields and adds these DIPOLE-specific settings:

```yaml
algorithm:
  type: "dipole"
  task_name: ${env.environment}            # used to look up BCE threshold + as G scoring key
  dipole:
    beta: 2.0
    k: 0.0
    guidance_omega: 2.0
    g_sign: "negate_raw"               # negate_raw | raw
    g_normalization: "batch_zscore"    # batch_zscore | running_zscore | minmax | none
    g_clip: 10.0
    polarity_embedding_init: "small_gaussian"   # small_gaussian | zero_neg | antipodal
    polarity_embedding_init_scale: 1.0e-3
    lpb_detector:
      ckpt_path: "checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth"   # MUST be set
      device: ${algorithm.flow.device}
      camera_to_view: {}              # {policy_camera_name: encoder_view_name}
  trainer:
    pretrain_steps: 0                  # DIPOLE always boots from runtime.init_checkpoint
runtime:
  init_checkpoint: null                # MUST be set
intervention:
  enabled: true                        # default-on
```

---

## Logged metrics (per train step)

Beyond standard flow-matching losses, DIPOLE emits:

- `raw_lpb_score_{mean,std,min,max}` — BCE failure score before sign flip
- `G_{mean,std}` — post-sign-flip + normalization
- `logit_{mean,std}` — `beta*G + k` after clamp
- `w_pos_{mean,std}`, `w_neg_{mean,std}`
- `frac_w_pos_saturated_high` (`>0.99`), `frac_w_pos_saturated_low` (`<0.01`)
- `loss_pos`, `loss_neg`, `loss_total`, plus per-branch flow/endpoint/smooth losses
- `frac_intervention`
- `polarity_embedding_{pos,neg}_norm`, `polarity_embedding_cos`

---

## Usage

1. Pre-fit a BCE LPB detector via the upstream benchmark runner. The shared 4-task checkpoint is already at `checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth`. To re-fit:

```bash
MODEL_CKPT=/abs/path/to/checkpoints/model_49.pth \
SAVE_CKPT_DIR=/abs/path/to/out/checkpoints \
bash robosuite/discriminator/lpb_v2/scripts/run_bce_robosuite_benchmark.sh
```

The runner saves `bce_head.pth` under `${SAVE_CKPT_DIR}`.

2. Train DIPOLE with GUI + intervention + online updates:

```bash
INIT_CHECKPOINT=/abs/flow_dagger_run/checkpoints/latest.pt \
LPB_CKPT=checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth \
BETA=2 K=0 OMEGA=2 \
bash robosuite/pipeline/scripts/train_dipole.sh
```

3. Sweep `omega` at eval time without retraining:

```bash
DIPOLE_MODE=eval OMEGA_RUNTIME_OVERRIDE=4 \
INIT_CHECKPOINT=... LPB_CKPT=... \
bash robosuite/pipeline/scripts/train_dipole.sh
```

---

## Known caveats / future work

- **LPB encoder runs per batch**: cost ≈ policy forward. A G cache keyed by `(transition_id, action_window_hash)` is the obvious v2 optimization.
- **CFG-at-start jitter**: see the caveat under "Polarity-conditioned policy" above. Levers if it shows up empirically: `polarity_embedding_init=zero_neg`, or add a `guidance_omega_warmup_steps` config and ramp ω 0→target.
- **Single-view encoders**: if the BCE ckpt's LPB encoder was trained on a single view (e.g. `view_names=["agentview"]`) but the policy uses several cameras, the provider only routes the matching view to the encoder. The other policy cameras are still used by the actor; they're just not seen by G.
- **Backbone drift**: because backbone is trainable and the polarity embedding is tiny, sustained sigmoid saturation will let the two branches re-converge. Diagnostics: `frac_w_pos_saturated_*` and `polarity_embedding_cos` in logs; lever is `beta` or switching to `running_zscore`.
- **Flow-Dagger internals**: the standalone Flow-Dagger train/eval entry points were removed from `robosuite.pipeline`, but DIPOLE still reuses its internal flow config, replay-buffer base, and image preprocessing helpers.

---

## Key reuse points (file:line)

- Flow policy backbone: `robosuite/policy/flow_multi/model.py:22-176`
- Flow image helpers: `robosuite/pipeline/algorithms/flow_dagger/models/flow.py`
- Flow replay-buffer base: `robosuite/pipeline/algorithms/flow_dagger/replay_buffer.py`
- LPB v2 encoder API: `discriminator/lpb_v2/detectors/single_bank_knn.py:106-399`
- BCEDiscriminator load/score: `discriminator/lpb_v2/detectors/bce.py:380-455`
