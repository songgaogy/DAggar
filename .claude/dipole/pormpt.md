# Coding Agent Prompt: Integrate DIPOLE into `robosuite.pipeline`

You are working in `/home/dodo/Documents/DAggar/robosuite`.

Your task is to integrate the DIPOLE algorithm into the existing `robosuite/pipeline` training stack, using LPB v2 from `robosuite/discriminator/lpb_v2` to provide the scalar `G` used by DIPOLE weighted behavior cloning(usage will be described in detail later in [how to get G](#lpb-v2-g-requirement)

## Required Context to Read First

Before proposing or editing code, read these files and inspect the implementation they point to:

- `.claude/dipole/2601.00898v2.pdf`
- `.claude/dipole/DIPOLE.md`
- `robosuite/pipeline/README.md`
- `robosuite/discriminator/lpb_v2/README.md`
- Existing pipeline algorithms:
  - `robosuite/pipeline/algorithms/flow_dagger/`
  - `robosuite/pipeline/algorithms/awr/`
  - `robosuite/pipeline/train_flow_dagger.py`
  - `robosuite/pipeline/train_awr.py`
  - `robosuite/pipeline/config/train_flow_dagger.yaml`
  - `robosuite/pipeline/config/train_awr.yaml`
  - `robosuite/pipeline/scripts/train_flow_dagger.sh`
  - `robosuite/pipeline/scripts/train_awr.sh`
- LPB v2 public APIs and detector outputs:
  - `robosuite/discriminator/lpb_v2/__init__.py`
  - `robosuite/discriminator/lpb_v2/detectors/`
  - `robosuite/discriminator/lpb_v2/adapters/`

Do not assume any local API exists. Search the workspace with `rg` and inspect source signatures before using classes, functions, config fields, checkpoint formats, or replay-buffer fields.

## Goal

Add a DIPOLE pipeline algorithm that can train a policy with bounded dichotomous weighted BC:

```text
positive weight: w_pos = sigmoid(beta * G + k)
negative weight: w_neg = 1 - w_pos
```

The implementation must support a base policy plus two trainable branches/adapters:

```text
positive branch: predicts action/noise/velocity optimized with w_pos
negative branch: predicts action/noise/velocity optimized with w_neg
```

At inference time, combine both branches using DIPOLE/CFG guidance:

```text
pred_guided = (1 + omega) * pred_pos - omega * pred_neg
```

Prefer reusing existing Flow-DAgger / flow-policy infrastructure if it already provides the right diffusion or flow matching policy abstraction. Only introduce new abstractions when the existing code cannot cleanly express DIPOLE.

## LPB v2 `G` Requirement

DIPOLE requires `G` to be larger for better actions/states. LPB v2 failure detectors often produce continuous scores where larger means more failure-like. Therefore, you must explicitly define and implement the sign convention:

```text
G = normalized_expert_or_success_score
```

However, if this transition data is from human intervention, make sure its weight to be 1 (G be positive infinite).

Requirements:

- Inspect the exact LPB v2 detector output structure before coding.
- Do not silently use raw LPB failure scores as positive `G`.
- Add config fields for selecting the LPB v2 source, checkpoint, feature source, transformer layer, score normalization, clipping, and sign convention.
- Implement deterministic normalization for `G`, such as batch z-score, running mean/std, min-max with configured bounds, or an existing local normalization utility. Expose the choice in Hydra config.
- Clamp or otherwise stabilize `beta * G + k` before sigmoid if needed to avoid numerical issues.
- Log summary statistics of raw LPB score, normalized `G`, `w_pos`, and `w_neg`.

## Expected Integration Shape

Actually you CAN do large refactor and modification under `robosuite/pipeline`, since other algorithms have been mainly stored in another branch, and this current `dipole` branch is used only for dipole. But you don't need to delete other algorothm.

## Training Behavior

Implement DIPOLE training as behavior cloning over replay/demo data with LPB-derived `G` weights.

For each sampled batch:

1. Load observations, actions, and any fields needed by the existing policy loss.
2. Compute or retrieve LPB v2 scores for the batch.
3. Convert LPB score to normalized `G` with the configured sign convention.
4. Compute:

   ```python
   w_pos = torch.sigmoid(beta * G + k)
   w_neg = 1.0 - w_pos
   ```

5. Compute positive weighted BC / diffusion / flow loss.
6. Compute negative weighted BC / diffusion / flow loss.
7. Optimize only the intended trainable parameters.
8. Log loss and weight statistics.

If the policy is a diffusion or flow policy, preserve the existing timestep/noise sampling convention from the pipeline implementation. Do not invent a new diffusion process if an existing one is already used.

## Inference Behavior

Expose `omega` as a Hydra config field and runtime override. At action selection time:

```python
pred_guided = (1.0 + omega) * pred_pos - omega * pred_neg
```

Preserve existing action clipping, scaling, device placement, image observation handling, locks, and asynchronous runtime behavior.

If existing policy code cannot produce separate positive and negative predictions at inference, stop and explain the exact blocker before doing a broad rewrite.

## Config and Script Requirements

Use Hydra config consistent with existing pipeline configs. Include at least:

- random seed
- dataset / replay input path
- base policy checkpoint or initialization path
- LPB v2 checkpoint path
- LPB v2 detector type (`bce`, `single_bank_knn`, or `two_bank_knn`, if supported)
- LPB v2 feature source and transformer layer
- `beta`
- `k`
- `omega`
- `g_sign`
- `g_normalization`
- `g_clip`
- batch size, learning rate, optimizer settings
- logging options
- checkpoint output directory
- device settings

Create a bash entrance under `robosuite/pipeline/scripts/` because this run requires long CLI parameters. Follow the existing script style and use:

```bash
export WANDB_MODE=offline
export WANDB_ENTITY=songgao-personal
```

Use `/home/dodo/miniconda3/envs/dagger/bin/python` as the default Python binary unless the local script conventions indicate otherwise.

communicate with me in Chinese. Ask me anything if you are unclear.