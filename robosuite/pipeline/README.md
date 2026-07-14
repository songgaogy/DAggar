# DIPOLE Pipeline

This package contains the simulation, data, discriminator, VAST value-stitching,
and DIPOLE policy-training pipeline. The discriminator is fixed; current RL work
uses the local **VAST value-stitching adaptation** described in
[`VAST_VALUE_STITCHING_ADAPTATION.md`](./VAST_VALUE_STITCHING_ADAPTATION.md).

The implementation is intentionally not presented as the complete upstream VAST
agent. It keeps the existing DIPOLE dual-branch policy and routed-sigmoid branch
weights, and uses VAST-style `G + target-V` stitching as the value signal.

## Algorithms

- `algorithms/vast/`: VAST configuration, replay, networks, joint G/V learner,
  warmup, tests, and visualization.
- `algorithms/dipole/`: dual-branch flow policy, advantage provider, replay, and
  trainer.
- `algorithms/discriminator/`: frozen nnPU discriminator and shared dynamics
  encoder.
- `offline/`: collected-data assembly, VAST finetuning, stitched-advantage
  precomputation, weighted behavior cloning, evaluation, and diagnostics.

The upstream reference source and paper remain unchanged under
`docs/vast_source-code/` and `docs/vast_paper.pdf`.

## Required checkpoints

- A pretrained flow-policy checkpoint for environment and policy metadata.
- A task-calibrated nnPU discriminator checkpoint.
- A schema-v7 VAST warmup checkpoint for new offline DIPOLE runs.

Tensor computation must run on CUDA. Launchers use the `dagger` conda environment
by default and log to TensorBoard.

## Quick start

### Initialize VAST offline

```bash
NNPU_CKPT=/path/to/pu_bce_head.pth \
CUDA_VISIBLE_DEVICES=1 \
bash robosuite/pipeline/scripts/utils/init_vast.sh
```

Default VAST warmup settings are `single_vast`, `K=10`, composition coefficient
`0.5`, expectile `0.9`, batch size `512`, and 30,000 joint G/V updates. The
launcher writes:

- `vast_state.pt`
- `tensorboard/`
- `data/<task>/offline_data-vast/vast_offline_transitions.pt`

The main controls are `JOINT_STEPS`, `BATCH_SIZE`, `VAST_V_MODE`, `VAST_MAX_K`,
`VAST_COMP_COEF`, `VAST_SAMPLING_SEED`, `EXPECTILE_TAU`, `G_LR`, `V_LR`,
`OUTPUT_REWARD_COEF`, `DISC_REWARD_COEF`, `OUTPUT_FILE`, and `SAVE_DIR`.

### Visualize a warmup checkpoint

```bash
VAST_CKPT=/path/to/vast_state.pt \
SPLIT=success \
bash robosuite/pipeline/scripts/utils/vis_init_vast.sh
```

The visualizer exports `steps.csv`, value/VAST plots, rollout video,
current/future-frame video, discriminator diagnostics, and `summary.json`.
`IQL_CKPT` and `--iql-ckpt` are accepted only as deprecated read aliases; setting
both the old and new names is an error.

### Train DIPOLE-RL

```bash
INIT_CHECKPOINT=/path/to/flow.pt \
NNPU_CKPT=/path/to/pu_bce_head.pth \
VAST_WARMUP_CKPT=/path/to/vast_state.pt \
bash robosuite/pipeline/scripts/train_dipole_rl.sh
```

The canonical Hydra namespace is `algorithm.vast`. The deprecated
`IQL_WARMUP_CKPT` launcher alias remains read-only for existing commands and
conflicts with `VAST_WARMUP_CKPT` when both are set.

## VAST value-stitching behavior

For action horizon `H`, the replay samples a macro distance `k` without crossing
an episode boundary or terminal. `G(s_t, s_{t+kH}, k)` is trained with a
discounted Monte-Carlo target and the VAST composition identity. The V target is

```text
stopgrad(G(s_t, s_{t+kH}, k)
         + gamma^(kH) * (1 - done) * V_target(s_{t+kH}))
```

and V uses expectile regression. `single_vast` is the default; `ensemble_lcb`
is retained as an ablation. The shared dynamics encoder and nnPU discriminator
are frozen.

Offline Phase B uses

```text
A_stitch = G(s, s_k, k) + gamma^(kH) V_target(s_k) - V(s)
```

with the existing routed-sigmoid weights. Windows with fewer than two remaining
macro chunks fall back to TD1. TD1 and GAE remain diagnostic signals, not the
default Phase-B value signal.

## Configuration

The main configuration is `config/train_dipole_rl.yaml`:

```yaml
algorithm:
  vast:
    enabled: true
    config:
      vast_v_mode: single_vast
      vast_max_k: 10
      vast_comp_coef: 0.5
      expectile_tau: 0.9
      device: cuda:0
    warmup_joint_steps: 15000
    warmup_ckpt: null
```

`config/train_offline_dipole.yaml` adds `offline.vast_finetune`,
`offline.vast_finetuned_path`, and `offline.vast_warmup_transitions_dir`.

## Checkpoint compatibility

New checkpoints use schema v7 and the canonical VAST state key and filenames.
Loaders may accept an existing schema-v6 checkpoint with the legacy state key as
a read-only compatibility path, but new writes always use VAST names. Schema-v5
and older value-only checkpoints cannot initialize VAST and must be regenerated
with `init_vast.sh`.

## Outputs and logging

- Warmup metrics use `vast/*` and `train/vast/*` TensorBoard tags.
- Offline Phase A uses `vast_value_stitching_finetune/*`.
- `run_info.json` records VAST checkpoint paths, buffer statistics, V mode,
  macro horizon, sampling seed, composition coefficient, and fallback fraction.
- The method label is `vast_value_stitching_adaptation` to distinguish this
  implementation from complete upstream VAST.

## Troubleshooting

- CUDA unavailable: stop and restore GPU access; there is no CPU tensor fallback.
- Schema mismatch: regenerate the warmup checkpoint with `init_vast.sh`.
- Configuration mismatch: use the V mode, K, horizon, projector dimensions, and
  reward coefficients recorded in the checkpoint.
- Slow warmup: use the pre-encoded replay cache and select a cache device that
  fits the dataset without changing sampling, loss, or model semantics.
