# Discriminator-weighted Offline DIPOLE

Offline DIPOLE trains the existing dual-branch policy from fixed, collected
episodes. It does not train a value function. The dynamics encoder and finetuned
nnPU discriminator are loaded once, frozen, and used to precompute one raw
score for every valid policy window.

The score convention matches the finetuned discriminator visualizations:
`s = raw_score = -head_logit`, where larger values are more failure-like. For
the checkpoint threshold in the same score space, the policy-window value and
branch weights are

```text
G       = threshold - s
w_pos   = sigmoid(beta * G)
w_neg   = 1 - w_pos
```

The checkpoint already contains folded logit normalization. Offline DIPOLE
therefore uses the raw head output directly and does not apply another
normalization. `beta` is configurable and defaults to `2`; the threshold always
comes from the checkpoint.

## Pipeline

### 1. Collect fixed-policy episodes

`scripts/collect_data.sh` runs the fixed policy and discriminator with optional
human intervention. It does not update the policy, encoder, or discriminator.
The output is `offline_episodes.pt` under the configured task data directory.

```bash
TASK=PickPlaceCereal \
bash robosuite/pipeline/offline/scripts/collect_data.sh
```

### 2. Finetune the discriminator

`scripts/finetune_disc.sh` warm-starts the existing nnPU head. The dynamics
encoder stays frozen, and the script never initializes a replacement head.
Collected policy-only sections form the offline positive/unlabeled pools;
human-executed and counterfactual actions are excluded from discriminator
training.

```bash
NNPU_CKPT=/path/to/pu_bce_head.pth \
OFFLINE_EPISODES=data/PickPlaceCereal/offline_data/offline_episodes.pt \
PRETRAIN_DIR=data/PickPlaceCereal/discriminator-pretrain-quadratic-c2-l1e2-v2 \
TASK=PickPlaceCereal \
bash robosuite/pipeline/offline/scripts/finetune_disc.sh
```

The standalone Hydra config is `pipeline/config/discriminator.yaml`.
TensorBoard is the only experiment logger. The default output root is
`outputs/dipole-rl-offline_disc`, with one run laid out as

```text
outputs/dipole-rl-offline_disc/<task>_<timestamp>[_suffix]/
└── discriminator/
    ├── checkpoints/pu_bce_head_finetuned.pth
    ├── run_info.json
    ├── config_resolved.yaml
    └── tensorboard/
```

The checkpoint preserves the nnPU schema and records finetuning provenance,
training history, pool statistics, folded normalization, and the recalibrated
threshold.

### 3. Train Offline DIPOLE

`scripts/train_offline_dipole.sh` takes the pipeline run directory as its
input. Its default is

```text
outputs/dipole-rl-offline_disc/PickPlaceCereal_20260719_212234
```

Run the policy stage with

```bash
PIPELINE_RUN_DIR=/path/to/pipeline_run \
bash robosuite/pipeline/offline/scripts/train_offline_dipole.sh
```

The launcher resolves exactly
`${PIPELINE_RUN_DIR}/discriminator/checkpoints/pu_bce_head_finetuned.pth`. It
does not search for a latest run and does not fall back to a parent nnPU
checkpoint. Policy checkpoints are written under
`<pipeline_run>/checkpoints/`; TensorBoard logs go to
`<pipeline_run>/tensorboard/dipole/`. An existing `checkpoints/` directory is
rejected.

The training entry point performs the following steps:

1. Load the flow policy and derive the action horizon from it.
2. Load and freeze the dynamics encoder and finetuned discriminator.
3. Convert `offline_episodes.pt` into routed policy, human, and counterfactual
   streams, and add clean pretrain demonstrations.
4. Build the static valid-window cache and precompute raw discriminator scores
   on CUDA in batches.
5. Train the dual-branch policy with weighted behavior cloning.

Routing semantics are:

| Route | Source | `(w_pos, w_neg)` |
| --- | --- | --- |
| `disc_weighted` | policy rollout action | `(sigmoid(beta * G), 1 - sigmoid(beta * G))` |
| `pos_only` | human/expert action | `(1, 0)` |
| `neg_only` | intervention-time policy action | `(0, 1)` |

The score cache is computed from unaugmented static observation windows. Image
augmentation, when enabled for policy training, does not change cached scores.

Important launcher settings are:

| Variable | Meaning |
| --- | --- |
| `PIPELINE_RUN_DIR` | Required pipeline run root (contains `discriminator/`); has the canonical default above |
| `BRANCH_BETA` | Sigmoid slope, default `2` |
| `POLICY_BATCH_SIZE` | Weighted-BC batch size |
| `PREENCODE_BATCH_SIZE` | CUDA discriminator precompute batch size |
| `NUM_TRAIN_STEPS` | Number of policy update steps |
| `USE_ONLINE_SUCCESS` | Include pure on-policy success episodes when enabled |

The policy-stage metadata identifies the algorithm as
`discriminator_weighted_offline_dipole`. TensorBoard and console diagnostics
include raw scores, the checkpoint threshold, `G`, both branch weights, and
route ratios.

## Evaluation and discriminator visualization

`scripts/eval_disc_finetuned.sh` evaluates a parent or finetuned nnPU checkpoint
on the configured validation data and collected offline episodes. It can also
generate the sampled discriminator visualization bundle.

```bash
FINETUNED_CKPT=/path/to/pu_bce_head_finetuned.pth \
MODEL_CKPT=/path/to/model_10.pth \
TASK=PickPlaceCereal \
bash robosuite/pipeline/offline/scripts/eval_disc_finetuned.sh
```

Set `GENERATE_VISUALS=False` for quantitative reports only, or `RUN_EVAL=False`
to regenerate only visualizations. All tensor inference is CUDA-only.

`scripts/eval_offline_dipole.sh` evaluates a trained policy checkpoint. Formal
policy training and evaluation runs should be launched only after their exact
configuration, seeds, metrics, compute, output directory, and stopping
conditions have been approved.

## Output layout

```text
<pipeline_run>/
├── discriminator/
│   ├── checkpoints/pu_bce_head_finetuned.pth
│   ├── run_info.json
│   ├── config_resolved.yaml
│   └── tensorboard/
├── checkpoints/
├── run_info.json
├── config_resolved.yaml
└── tensorboard/
    └── dipole/
```

No VAST/value-function checkpoint is created by the offline pipeline. Online
DIPOLE-RL and the shared `algorithms/vast/` implementation remain unchanged.

## File map

```text
offline/
├── scripts/
│   ├── collect_data.sh
│   ├── finetune_disc.sh
│   ├── eval_disc_finetuned.sh
│   ├── train_offline_dipole.sh
│   └── eval_offline_dipole.sh
├── src/
│   ├── finetune_disc.py
│   ├── eval_disc_checkpoint.py
│   ├── visualize_disc_finetuned.py
│   ├── train_offline_dipole.py
│   └── eval_offline_dipole.py
├── discriminator/
├── visualization/
└── utils/
    ├── discriminator_scores.py
    ├── branch_weights.py
    ├── buffer.py
    └── episode_dataset.py
```
