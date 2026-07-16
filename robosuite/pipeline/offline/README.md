# Offline DIPOLE with VAST Value Stitching

Offline DIPOLE has three stages: collect fixed-policy episodes, finetune the VAST
G/V learner, then train the existing dual-branch DIPOLE policy with GAE computed
from the finetuned VAST V / target-V pair. The nnPU discriminator and shared
dynamics encoder remain frozen inside Offline DIPOLE. A separate, optional nnPU
warm-start workflow updates the discriminator before it is supplied to another
run; it is not invoked by `train_offline_dipole.sh`.

## Prerequisite: VAST warmup

Run `scripts/utils/init_vast.sh` before offline training. It produces a schema-v7
(`single_vast`) or schema-v8 (`indep_ensemble`) `vast_state.pt` and
`data/<task>/offline_data-vast/vast_offline_transitions.pt`.
The offline run must use the same action horizon, encoder metadata, V mode, K,
reward coefficients, and sampling semantics.

## Stage 1: collect episodes

`scripts/collect_data.sh` runs the fixed policy and discriminator with optional
human intervention. It does not update the policy, VAST learner, encoder, or
discriminator. The output is `offline_episodes.pt` under the configured task data
directory.

## Independent nnPU discriminator update

The discriminator update is a standalone CUDA-only workflow. It freezes the
dynamics encoder, uses the checkpoint-selected transformer layer and action
chunk contract, and continues optimizing the existing nnPU head weights. It
never initializes a replacement head.

### Extract the original pretrain pools

`discriminator/dyn_disc/scripts/extract_pretrain_data.sh` reconstructs the
original success split from the parent checkpoint metadata. For the canonical
PickPlaceCereal run, `seed=0`, `calib_fraction=0.2`, and a 50-trajectory cap
produce 40 `positive_train` and 10 `positive_calib` trajectories. Whole failure
rollouts form `unlabeled_train`.

```bash
NNPU_CKPT=/path/to/pu_bce_head.pth \
TASK=PickPlaceCereal \
bash robosuite/discriminator/dyn_disc/scripts/extract_pretrain_data.sh
```

The default output is
`data/<task>/discriminator-pretrain/{manifest.json,positive_train/,positive_calib/,unlabeled_train/}`.
Each atomic `.pt` shard stores float32 latents, frame indices, and provenance;
the manifest pins both the dynamics model and saved normalizer by SHA-256.
Existing output is rejected unless overwrite is explicitly requested.

### Warm-start finetuning

`offline/scripts/finetune_disc.sh` keeps the extracted replay pools separate from
the collected GT pools.
Collected episodes are split at intervention boundaries and episode ends. Only
non-intervention `executed_action` rows are encoded, after checking equality with
`policy_action`; human actions and counterfactual actions are excluded.

A policy section enters the positive pool only when its final executed action
directly causes task success. Success completed by a human section produces no
positive section. Every other policy section enters the unlabeled pool,
including intervention-ending, reset, max-step, environment-done, and
interrupted sections. Short sections are retained, and action chunks repeat-pad
within their section without crossing a boundary.

```bash
NNPU_CKPT=/path/to/pu_bce_head.pth \
OFFLINE_EPISODES=data/PickPlaceCereal/offline_data/offline_episodes.pt \
PRETRAIN_DIR=data/PickPlaceCereal/discriminator-pretrain \
TASK=PickPlaceCereal \
bash robosuite/pipeline/offline/scripts/finetune_disc.sh
```

The standalone Hydra config is `pipeline/config/finetune_disc.yaml`. Launcher
environment variables expose the parent checkpoint and optional encoder
override, CUDA device, input paths, schedule, seed, run root/suffix, logging
interval, `LAMBDA_PRE`, `LAMBDA_P`, `LAMBDA_N`, and the two GT batch sizes.
The positive safety-margin controls are exposed as `SAFETY_MARGIN_WEIGHT`,
`SAFETY_MARGIN_DELTA`, `SAFETY_MARGIN_TEMPERATURE`, and
`SAFETY_MARGIN_BOUNDARY_SOURCE`.
Defaults are AdamW with cosine decay for 10 epochs, learning rate `3e-5`, weight
decay `1e-4`, and seed `0`. TensorBoard is the only experiment logger.

The objective uses three independently sampled risks:
`1.0 * L_pre-nnPU + 0.025 * (L_P_BCE + L_P_safe) + 0.0125 * L_N`. The safety
term is `softplus((m_k + 1.0 - g) / 1.0)`, where `m_k=-tau_k` is frozen from
the calibrated parent checkpoint. Pretrain replay draws 256
frames each from the original positive and unlabeled pools. `L_P` draws 256
frames from direct-success offline policy segments, while `L_N` draws 256
frames from intervention GT windows. Offline success frames never enter the
pretrain positive pool or the nnPU risk. Threshold calibration uses only the
held-out original `positive_calib` split. The standalone command accepts a
single-task parent checkpoint; a multi-task shared head is rejected because
every task threshold would need its own held-out recalibration pool.

Each run writes
`outputs/discriminator-finetune/<task>_<timestamp>[_suffix]/` with:

- `checkpoints/pu_bce_head_finetuned.pth`, compatible with
  `FrozenNNPUDiscriminator`
- `run_info.json` and `config_resolved.yaml`
- TensorBoard events under `tensorboard/`

The checkpoint preserves the original top-level nnPU schema and adds provenance,
training history, pool statistics, and recalibration metadata.

### Visualize the finetuned head

The visualization launcher loads only the finetuned checkpoint and runs the
pipeline-owned finetuned visualizer on deterministic samples from
`fail_rollout-val-labeled`, `success_rollout-val`, and the collected offline
episodes. It also samples `offline-success` episodes that terminated successfully
without any intervention frames:

```bash
FINETUNED_CKPT=/path/to/pu_bce_head_finetuned.pth \
MODEL_CKPT=/path/to/model_10.pth \
TASK=PickPlaceCereal \
bash robosuite/pipeline/offline/scripts/vis_disc_finetuned.sh
```

It writes one MP4 per trajectory plus `finetuned_scores.pdf`,
`finetuned_scores_offline.pdf`, and `finetuned_scores_offline-success.pdf` under
the finetune run's
`visualization/val-seed<seed>/` directory by default. This command does not fit
a head or run benchmark evaluation.

## Stage 2: offline training

### Phase A: joint VAST G/V finetuning

Phase A mixes collected `policy_bc` transitions with the warmup transitions and
continues the same joint G/V update used by warmup. It does not add human,
negative, online-success, or pretrain streams to the VAST finetune buffer.

`SKIP_RL=1` requires a compatible finetuned VAST checkpoint. Legacy
schema-v5 value-only states are rejected.

### Phase B: routed weighted behavior cloning

Each valid policy window first receives the one-macro-step TD residual

```text
delta_t = r_chunk(t) + gamma^H * m_t * V_target(s_{t+H}) - V(s_t)
A_t     = delta_t + gamma^H * lambda * m_t * A_{t+H}
w_pos   = sigmoid(beta * (A_t + k_offset))
w_neg   = 1 - w_pos
```

Here `m_t=1-done_t`. The recursive future term is zero when no valid same-section
`t+H` successor row exists. The recursion uses `lambda=0.6` by default. Human
and negative routes keep their existing fixed branch semantics.
Set the estimator explicitly to `td1` for the one-macro-step ablation. Stitched
advantages remain available for VAST diagnostics but are not Phase-B weights.

## Run

```bash
POLICY_CKPT=/path/to/flow.pt \
NNPU_CKPT=/path/to/pu_bce_head.pth \
VAST_CKPT=/path/to/vast_state.pt \
bash robosuite/pipeline/offline/scripts/train_offline_dipole.sh
```

Canonical launcher variables are `VAST_CKPT` and `VAST_FINETUNED`. Deprecated
IQL-named variables are read-only aliases; providing both old and new forms is an
error.

Key Hydra settings are:

| Setting | Meaning |
| --- | --- |
| `algorithm.vast.warmup_ckpt` | schema-v7/v8 warmup checkpoint; schema-v6 is read-only compatible |
| `algorithm.vast.config.*` | VAST architecture, reward, optimizer, and sampling settings |
| `offline.vast_warmup_transitions_dir` | warmup transition directory under `data/<task>` |
| `offline.vast_finetuned_path` | checkpoint used when Phase A is skipped |
| `offline.vast_finetune.{num_steps,batch_size,preencode_cache}` | Phase-A schedule |
| `offline.advantage.estimator` | `gae` by default; set `td1` for the explicit ablation |
| `offline.advantage.gae_lambda` | GAE lambda, default `0.6` |
| `offline.branch_weight.{beta,k}` | routed-sigmoid slope and offset |

## Outputs

Each run writes:

- policy checkpoints under `checkpoints/`
- `checkpoints/vast_state_finetuned.pt`
- TensorBoard events under `tensorboard/`
- `run_info.json` and the resolved configuration
- branch-weight diagnostics; the VAST visualizer provides GAE, TD1, and stitched diagnostics

The run metadata identifies the algorithm as
`vast_value_stitching_adaptation`, records the VAST checkpoint paths and buffer
statistics, and includes the Phase-B advantage estimator and GAE lambda.

## Visualization

Use `offline/scripts/vis_vast_finetuned.sh` for a finetuned checkpoint. Each run
exports `steps.csv`, stitched-value plots, rollout video, current/future-frame
video, discriminator CSV/plots/HUD video, and `summary.json`.

Visualization is CUDA-only and accepts schema-v7/v8 checkpoints. Existing
schema-v6 artifacts with legacy payload naming may be read through the explicit
compatibility path, but all new outputs use VAST names.

Use `offline/scripts/vis_disc_finetuned.sh` for the independent nnPU output, as
described above. Both visualization launchers reject non-CUDA devices.

## File map

```text
offline/
├── scripts/
│   ├── collect_data.sh
│   ├── finetune_disc.sh
│   ├── train_offline_dipole.sh
│   ├── vis_disc_finetuned.sh
│   └── vis_vast_finetuned.sh
├── src/
│   ├── finetune_disc.py
│   ├── visualize_disc_finetuned.py
│   ├── train_offline_dipole.py
│   ├── eval_offline_dipole.py
│   └── diagnostic_plots.py
├── discriminator/
│   ├── episodes.py
│   ├── features.py
│   ├── pools.py
│   ├── encoder.py
│   ├── contracts.py
│   ├── trainer.py
│   └── checkpoint.py
├── visualization/
│   ├── disc_adapter.py
│   ├── disc_contract.py
│   ├── disc_episodes.py
│   └── disc_renderer.py
└── utils/
    ├── vast_finetune.py
    ├── advantage.py
    ├── branch_weights.py
    ├── buffer.py
    └── episode_dataset.py
```
