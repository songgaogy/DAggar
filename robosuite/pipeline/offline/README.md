# Offline DIPOLE with VAST Value Stitching

Offline DIPOLE has three stages: collect fixed-policy episodes, finetune the VAST
G/V learner, then train the existing dual-branch DIPOLE policy with stitched
advantages. The nnPU discriminator and shared dynamics encoder remain frozen.

## Prerequisite: VAST warmup

Run `scripts/utils/init_vast.sh` before offline training. It produces a schema-v7
`vast_state.pt` and `data/<task>/offline_data-vast/vast_offline_transitions.pt`.
The offline run must use the same action horizon, encoder metadata, V mode, K,
reward coefficients, and sampling semantics.

## Stage 1: collect episodes

`scripts/collect_data.sh` runs the fixed policy and discriminator with optional
human intervention. It does not update the policy, VAST learner, encoder, or
discriminator. The output is `offline_episodes.pt` under the configured task data
directory.

## Stage 2: offline training

### Phase A: joint VAST G/V finetuning

Phase A mixes collected `policy_bc` transitions with the warmup transitions and
continues the same joint G/V update used by warmup. It does not add human,
negative, online-success, or pretrain streams to the VAST finetune buffer.

`SKIP_RL=1` requires a compatible finetuned VAST checkpoint. Legacy
schema-v5 value-only states are rejected.

### Phase B: routed weighted behavior cloning

Each valid policy window receives one deterministic, seed-controlled legal `k`:

```text
A_stitch = G(s, s_k, k) + gamma^(kH) V_target(s_k) - V(s)
w_pos    = sigmoid(beta * (A_stitch + k_offset))
w_neg    = 1 - w_pos
```

Windows with fewer than two remaining macro chunks use TD1 fallback. Human and
negative routes keep their existing fixed branch semantics. TD1 and GAE are
retained for diagnostics only.

## Run

```bash
INIT_CHECKPOINT=/path/to/flow.pt \
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
| `algorithm.vast.warmup_ckpt` | schema-v7 warmup checkpoint; schema-v6 is read-only compatible |
| `algorithm.vast.config.*` | VAST architecture, reward, optimizer, and sampling settings |
| `offline.vast_warmup_transitions_dir` | warmup transition directory under `data/<task>` |
| `offline.vast_finetuned_path` | checkpoint used when Phase A is skipped |
| `offline.vast_finetune.{num_steps,batch_size,preencode_cache}` | Phase-A schedule |
| `offline.advantage.estimator` | fixed to `vast_stitched` for Phase B |
| `offline.branch_weight.{beta,k}` | routed-sigmoid slope and offset |

## Outputs

Each run writes:

- policy checkpoints under `checkpoints/`
- `checkpoints/vast_state_finetuned.pt`
- TensorBoard events under `tensorboard/`
- `run_info.json` and the resolved configuration
- branch-weight and stitched-advantage diagnostics

The run metadata identifies the algorithm as
`vast_value_stitching_adaptation`, records the VAST checkpoint paths and buffer
statistics, and includes the sampled-horizon/fallback configuration.

## Visualization

Use `offline/scripts/vis_vast_finetuned.sh` for a finetuned checkpoint. Each run
exports `steps.csv`, stitched-value plots, rollout video, current/future-frame
video, discriminator CSV/plots/HUD video, and `summary.json`.

Visualization is CUDA-only and accepts schema-v7 checkpoints. Existing
schema-v6 artifacts with legacy payload naming may be read through the explicit
compatibility path, but all new outputs use VAST names.

## File map

```text
offline/
├── scripts/
│   ├── collect_data.sh
│   ├── train_offline_dipole.sh
│   └── vis_vast_finetuned.sh
├── src/
│   ├── train_offline_dipole.py
│   ├── eval_offline_dipole.py
│   └── diagnostic_plots.py
└── utils/
    ├── vast_finetune.py
    ├── advantage.py
    ├── branch_weights.py
    ├── buffer.py
    └── episode_dataset.py
```
