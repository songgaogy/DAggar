# Offline DIPOLE with VAST Value Stitching

Offline DIPOLE has four stages: collect fixed-policy episodes, finetune the nnPU discriminator, finetune the VAST G/V learner, then train the existing dual-branch DIPOLE policy with GAE computed from the finetuned VAST V / target-V pair. The finetuned nnPU discriminator and shared dynamics encoder remain frozen inside Offline DIPOLE. The discriminator and DIPOLE stages share one pipeline run directory, while their artifacts remain isolated in stage subdirectories.

## Prerequisite: VAST warmup

Run `scripts/utils/init_vast.sh` before offline training. It produces a schema-v7 (`single_vast`) or schema-v8 (`indep_ensemble`) `vast_state.pt` and `data/<task>/offline_data-vast/vast_offline_transitions.pt`. The offline run must use the same action horizon, encoder metadata, V mode, K, reward coefficients, and sampling semantics.

## Stage 1: collect episodes

`scripts/collect_data.sh` runs the fixed policy and discriminator with optional human intervention. It does not update the policy, VAST learner, encoder, or discriminator. The output is `offline_episodes.pt` under the configured task data directory.

## Independent nnPU discriminator update

The discriminator update is a CUDA-only workflow that starts a pipeline run. It freezes the dynamics encoder, uses the checkpoint-selected transformer layer and action chunk contract, and continues optimizing the existing nnPU head weights. It never initializes a replacement head.

### Extract the original pretrain pools

`discriminator/dyn_disc/scripts/extract_pretrain_data.sh` reconstructs the original success split from the parent checkpoint metadata. For the canonical PickPlaceCereal run, `seed=0`, `calib_fraction=0.2`, and a 50-trajectory cap produce 40 `positive_train` and 10 `positive_calib` trajectories. Whole failure rollouts form `unlabeled_train`.

```bash
NNPU_CKPT=/path/to/pu_bce_head.pth \
TASK=PickPlaceCereal \
bash robosuite/discriminator/dyn_disc/scripts/extract_pretrain_data.sh
```

The v2 default output is `data/<task>/discriminator-pretrain-quadratic-c2-l1e2-v2/{manifest.json,positive_train/,positive_calib/,unlabeled_train/}`. Each atomic `.pt` shard stores float32 latents, frame indices, and provenance. Cache reuse requires exact dynamics-model and normalizer SHA-256 values, feature contract, and ordered positive-train, positive-calibration, and unlabeled trajectory IDs. The source parent discriminator SHA-256 remains recorded for provenance but is not part of cache identity, because nnPU head weights and percentile thresholds do not affect frozen dynamics latents. Existing output is rejected unless overwrite is explicitly requested.

### Warm-start finetuning

`offline/scripts/finetune_disc.sh` keeps the extracted replay pools separate from the collected GT pools. Collected episodes are split at intervention boundaries and episode ends. Only non-intervention `executed_action` rows are encoded, after checking equality with `policy_action`; human actions and counterfactual actions are excluded.

A policy section enters the positive pool only when its final executed action directly causes task success. Success completed by a human section produces no positive section. Every other policy section enters the unlabeled pool, including intervention-ending, reset, max-step, environment-done, and interrupted sections. Short sections are retained, and action chunks repeat-pad within their section without crossing a boundary.

```bash
NNPU_CKPT=/path/to/pu_bce_head.pth \
OFFLINE_EPISODES=data/PickPlaceCereal/offline_data/offline_episodes.pt \
PRETRAIN_DIR=data/PickPlaceCereal/discriminator-pretrain-quadratic-c2-l1e2-v2 \
TASK=PickPlaceCereal \
bash robosuite/pipeline/offline/scripts/finetune_disc.sh
```

The standalone Hydra config is `pipeline/config/discriminator.yaml`. Launcher environment variables expose the parent checkpoint and optional encoder override, CUDA device, input paths, schedule, seed, run root/suffix, logging interval, `LAMBDA_PRE`, `LAMBDA_P`, `LAMBDA_N`, and the two GT batch sizes. `G_NORMALIZATION_ENABLED` toggles the fixed held-out-success normalization. Quadratic-cap controls are exposed as `QUADRATIC_CAP_ENABLED`, `QUADRATIC_CAP_C`, and `QUADRATIC_CAP_LAMBDA`. Defaults are AdamW for 20 epochs with a 20-epoch cosine horizon, learning rate `1e-5`, weight decay `1e-4`, seed `0`, normalization enabled, and weights `lambda_pre=0.1`, `lambda_P=0.2`, `lambda_N=0.01`. TensorBoard is the only experiment logger.

The objective uses three independently sampled risks: `lambda_pre * L_pre-nnPU + lambda_P * softplus(-g_P) + lambda_N * softplus(g_N)`, plus the v2 quadratic replay regularizer `0.01 * mean(relu(abs(g_pre)-2)^2)`. The cap sees only concatenated pretrain positive and unlabeled logits. With normalization enabled, all three risks and the cap use `g'=(g-m)/s`, where `m=-tau_parent` and `s=(Q75-Q25)/1.349` are computed once from the held-out `positive_calib` trajectories on CUDA. The transform is fixed during optimization, folded into the head before saving, and the final threshold is recalibrated on the same held-out pool. The pretrain replay batch contains 128 positive and 128 unlabeled frames; the offline-positive and GT-negative terms each draw 128 frames. Offline success frames never enter the replay nnPU risk. A multi-task shared head is rejected because every task requires its own held-out calibration contract.

The default weights were selected from the PickPlaceCereal seed-0 sweep because they improved the main validation metrics while satisfying all three loss-stability checks. This is not a claim of cross-seed or cross-task generalization. The selected run's offline-success frame FAR was `0.01751`, above the v2 parent's `0.00246`; keep that known trade-off visible when using the defaults.

Each finetune launch creates `outputs/dipole-rl-offline_vast/<task>_<timestamp>[_suffix]/` and writes the discriminator stage under `discriminator/` with:

- `discriminator/checkpoints/pu_bce_head_finetuned.pth`, compatible with `FrozenNNPUDiscriminator`
- `run_info.json` and `config_resolved.yaml`
- TensorBoard events under `tensorboard/`

The checkpoint preserves the original top-level nnPU schema and adds provenance, training history, pool statistics, and recalibration metadata.

### Evaluate and visualize the finetuned head

The evaluation launcher is the one-command entry point for the complete finetuned-checkpoint report. It evaluates all validation trajectories, every policy-only successful offline episode, and the exact GT-negative training frames reconstructed from checkpoint provenance. It then optionally generates deterministic sampled visualizations. Evaluation and visualization outputs remain separated:

```bash
FINETUNED_CKPT=/path/to/pu_bce_head_finetuned.pth \
MODEL_CKPT=/path/to/model_10.pth \
TASK=PickPlaceCereal \
bash robosuite/pipeline/offline/scripts/eval_disc_finetuned.sh
```

By default it writes:

- `evaluation/val-seed<seed>/benchmark.json`: full failure/success validation benchmark
- `evaluation/val-seed<seed>/success_false_alarm.json`: full validation-success false-alarm report
- `evaluation/val-seed<seed>/offline_success_false_alarm.json`: full offline-success false-alarm report
- `evaluation/val-seed<seed>/gt_fail_detection.json`: CUDA diagnostic on the exact GT-negative training frames; this is reported only and is not a held-out generalization result or hard gate
- `visualization/val-seed<seed>/finetuned_scores.pdf`, `finetuned_scores_offline.pdf`, and `finetuned_scores_offline-success.pdf`
- sampled MP4 files under `visualization/val-seed<seed>/videos/` for validation failure, validation success, offline, and offline-success trajectories

The default visualization sample count is `NUM_TRAJS=10` per pool, with `SPLIT=both` and `SEED=0`. Set `OUT_DIR` to override only the evaluation output directory and `VIS_OUT_DIR` to override only the visualization output directory. Both success reports exclude the first success frame and all later recorded frames. An offline episode enters the offline-success pool only when it has `terminal_reason=success` and contains no intervention.

Set `GENERATE_VISUALS=False` to run the four quantitative reports without PDFs or videos. Parent pretraining checkpoints write `gt_fail_detection.json` with `applicable=false` because they have no offline GT-negative training pool, and must be evaluated with visualization disabled:

```bash
GENERATE_VISUALS=False \
FINETUNED_CKPT=/path/to/parent_pu_bce_head.pth \
MODEL_CKPT=/path/to/model_10.pth \
TASK=PickPlaceCereal \
bash robosuite/pipeline/offline/scripts/eval_disc_finetuned.sh
```

To regenerate only the sampled visualization bundle without rerunning the full quantitative evaluation:

```bash
RUN_EVAL=False \
FINETUNED_CKPT=/path/to/pu_bce_head_finetuned.pth \
MODEL_CKPT=/path/to/model_10.pth \
TASK=PickPlaceCereal \
bash robosuite/pipeline/offline/scripts/eval_disc_finetuned.sh
```

## Stage 2: offline training

### Phase A: joint VAST G/V finetuning

Phase A mixes collected `policy_bc` transitions with the warmup transitions and continues the same joint G/V update used by warmup. It does not add human, negative, online-success, or pretrain streams to the VAST finetune buffer.

`SKIP_RL=1` requires a compatible finetuned VAST checkpoint. Legacy schema-v5 value-only states are rejected.

### Phase B: routed weighted behavior cloning

Each valid policy window first receives the one-macro-step TD residual

```text
delta_t = r_chunk(t) + gamma^H * m_t * V_target(s_{t+H}) - V(s_t)
A_t     = delta_t + gamma^H * lambda * m_t * A_{t+H}
w_pos   = sigmoid(beta * (A_t + k_offset))
w_neg   = 1 - w_pos
```

Here `m_t=1-done_t`. The recursive future term is zero when no valid same-section `t+H` successor row exists. The recursion uses `lambda=0.6` by default. Human and negative routes keep their existing fixed branch semantics. Set the estimator explicitly to `td1` for the one-macro-step ablation. Stitched advantages remain available for VAST diagnostics but are not Phase-B weights.

## Run the DIPOLE stage

```bash
PIPELINE_RUN_DIR=/path/to/outputs/dipole-rl-offline_vast/PickPlaceCereal_<timestamp> \
bash robosuite/pipeline/offline/scripts/train_offline_dipole.sh
```

`PIPELINE_RUN_DIR` is required. The launcher reads only `${PIPELINE_RUN_DIR}/discriminator/checkpoints/pu_bce_head_finetuned.pth`; it does not search for a latest run or fall back to the parent nnPU checkpoint. The DIPOLE stage writes to `${PIPELINE_RUN_DIR}/dipole/` and rejects an existing stage directory.

Canonical launcher variables are `VAST_CKPT` and `VAST_FINETUNED`. Deprecated IQL-named variables are read-only aliases; providing both old and new forms is an error.

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

The complete pipeline layout is:

```text
<pipeline_run>/
├── discriminator/
│   ├── checkpoints/pu_bce_head_finetuned.pth
│   ├── run_info.json
│   ├── config_resolved.yaml
│   └── tensorboard/
└── dipole/
    ├── checkpoints/
    ├── run_info.json
    ├── config_resolved.yaml
    └── tensorboard/
```

The DIPOLE stage writes:

- policy checkpoints under `checkpoints/`
- `checkpoints/vast_state_finetuned.pt`
- TensorBoard events under `tensorboard/`
- `run_info.json` and the resolved configuration
- branch-weight diagnostics; the VAST visualizer provides GAE, TD1, and stitched diagnostics

The run metadata identifies the algorithm as `vast_value_stitching_adaptation`, records the VAST checkpoint paths and buffer statistics, and includes the Phase-B advantage estimator and GAE lambda.

## Visualization

Use `offline/scripts/vis_vast_finetuned.sh` for a finetuned checkpoint. Each run exports `steps.csv`, stitched-value plots, rollout video, current/future-frame video, discriminator CSV/plots/HUD video, and `summary.json`.

Visualization is CUDA-only and accepts schema-v7/v8 checkpoints. Existing schema-v6 artifacts with legacy payload naming may be read through the explicit compatibility path, but all new outputs use VAST names.

Use `offline/scripts/eval_disc_finetuned.sh` with `RUN_EVAL=False` for the independent nnPU visualization bundle, as described above. Both visualization launchers reject non-CUDA devices.

## File map

```text
offline/
├── scripts/
│   ├── collect_data.sh
│   ├── finetune_disc.sh
│   ├── eval_disc_finetuned.sh
│   ├── train_offline_dipole.sh
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
