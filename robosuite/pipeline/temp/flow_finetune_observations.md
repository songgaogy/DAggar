# Flow Offline Finetune Observations

## Experiment Context

- Original policy: multitask flow checkpoint trained on 6 tasks, with 20 expert trajectories per task.
- Current experiment: warm-start from the original multitask checkpoint, then offline finetune on another 20 trajectories for one single task.
- Target task discussed: `PickPlaceBread`.
- Observed closed-loop success rate:
  - Original checkpoint: `0.68`
  - Finetune `500` steps: `0.18`
  - Finetune `2000` steps: `0.10`
  - Finetune `10000` steps: `0.22`

The key symptom is not only long-run overfitting. The policy collapses quickly after only `500` gradient steps, which suggests the finetune recipe is damaging deployable policy behavior early.

## Checked Code Path

Main script inspected:

- `robosuite/pipeline/temp/train_flow_offline.py`

Related implementation inspected:

- `robosuite/pipeline/temp/eval_flow_offline.py`
- `robosuite/pipeline/algorithms/flow_dagger/agent.py`
- `robosuite/pipeline/algorithms/flow_dagger/models/flow.py`
- `robosuite/pipeline/algorithms/flow_dagger/trainer.py`
- `robosuite/pipeline/algorithms/flow_dagger/replay_buffer.py`
- `robosuite/pipeline/utils/io.py`
- `robosuite/policy/flow_multi/train_flow.py`

## Code Observations

### 1. Original Checkpoint Uses EMA Weights

`FlowDaggerAgent.load_flow_policy_checkpoint()` loads:

```python
model_state = payload.get("ema_model", payload.get("model"))
```

So the original checkpoint evaluation is based on `ema_model` when available.

However, `train_flow_offline.py` saves finetuned checkpoints through:

```python
agent.save_checkpoint(...)
```

The saved `core` payload contains the current raw training model state, not a newly maintained EMA model. Therefore the comparison is effectively:

```text
original success 0.68: original EMA model
finetuned success: raw finetuned model
```

This is not an apples-to-apples deployment comparison.

### 2. Finetune Does Not Maintain EMA

Original `flow_multi` training maintains EMA:

```python
ema_model.update_parameters(model)
```

The offline finetune path does not maintain an EMA shadow model. For flow matching / diffusion-like policies, EMA can be important for stable rollout behavior. A raw model after several hundred steps can have lower training loss but worse closed-loop success.

### 3. Finetune Updates All Trainable Parameters

The finetune path does not restrict training to a small adaptation head. It constructs AdamW from all `requires_grad=True` parameters:

```python
self.optimizer = torch.optim.AdamW(
    [param for param in self.model.parameters() if param.requires_grad],
    lr=float(config.learning_rate),
    weight_decay=float(config.weight_decay),
)
```

This includes more than the flow head. Based on the checkpoint model config, trainable components can include:

- ResNet `layer3` and `layer4`
- visual token projections
- language projection layers
- fusion transformer
- aggregator
- flow head

For 20 trajectories, this is a large update surface.

### 4. ResNet BatchNorm Running Stats Are Likely Being Damaged

During update, the code calls:

```python
self.model.train(True)
```

This puts trainable ResNet stages into training mode, so BatchNorm running statistics in `layer3/layer4` are updated on the 20-demo finetune distribution.

A checkpoint comparison showed large BatchNorm buffer shifts after finetuning. Examples:

```text
image_encoder.backbone.layer4.1.bn2.running_mean rel_delta ~= 0.595
image_encoder.backbone.layer4.1.bn1.running_var  rel_delta ~= 0.541
image_encoder.backbone.layer4.0.bn2.running_var  rel_delta ~= 0.378
```

This is a strong candidate for the rapid success-rate collapse after only `500` steps. BatchNorm statistics can change quickly even if gradient updates are not huge.

### 5. Learning Rate Is Likely Too Aggressive For Warm-Start Finetune

The finetune uses:

```yaml
learning_rate: 1.0e-4
```

This is the original training LR, not a conservative finetune LR. With only 20 trajectories and full-model updates, `1e-4` is likely too large.

### 6. Demo Subset Sampling Is Not Reproducible

`train_flow_offline.py` calls:

```python
load_demo_paths(..., random_sample=True)
```

The underlying implementation uses:

```python
random.Random().sample(...)
```

This creates an unseeded RNG and ignores the global seed. Therefore the selected 20 demos may differ between runs even when `cfg.seed` is fixed.

This does not explain a single run collapse by itself, but it makes debugging and comparison unreliable.

### 7. Offline BC Data May Not Cover Closed-Loop Failure States

Even when the 20 extra trajectories are from the same task, they are still expert-state data. They may not cover states visited by the current policy after small mistakes.

So lower supervised imitation loss on expert trajectories does not guarantee higher closed-loop success. The finetuned policy can become more precise on the new demos but less robust to policy-induced off-manifold states.

## Most Likely Failure Causes

Ranked by current evidence:

1. **EMA mismatch:** original uses EMA weights, finetuned checkpoints use raw model weights.
2. **BatchNorm stat drift:** `model.train(True)` updates ResNet `layer3/layer4` BatchNorm running stats on only 20 demos.
3. **Full-model high-LR finetune:** too many modules are updated with `1e-4`.
4. **Offline BC limitation:** expert demos do not necessarily fix closed-loop failure states.
5. **Unseeded demo sampling:** results are not fully reproducible.

## Recommended Fixes

### Minimal Fixes To Try First

1. Maintain EMA during offline finetune and evaluate EMA weights.
2. Freeze visual BatchNorm running stats during finetune.
3. Lower finetune LR to `1e-5` or `3e-6`.
4. Restrict trainable scope first to `flow_head` only.
5. Fix demo subset selection with a seeded RNG and save selected demo names into checkpoint metadata.

### Suggested Finetune Matrix

Run short curves before long finetuning:

```text
steps: 0, 50, 100, 200, 500, 1000, 2000
lr:    1e-5, 3e-6
scope: flow_head
       flow_head + aggregator
BN:    frozen
EMA:   enabled
```

The important check is whether success improves before loss keeps decreasing.

### Suggested Eval Protocol

Use one eval script and identical rollout seeds for:

- original `flow_multi` checkpoint
- finetune `0` step checkpoint
- finetune `50/100/200/500/1000/2000` step checkpoints
- final checkpoint

Keep fixed:

- `max_steps`
- deterministic vs stochastic sampling
- environment metadata
- camera names
- task prompt
- episode seeds

The current `eval_flow_offline.py` should be extended to evaluate the original `flow_multi` checkpoint format directly, because it currently expects the finetuned `flow_dagger` checkpoint format with a `core` key.

## Practical Next Change Plan

If modifying code, the smallest useful change is to update `robosuite/pipeline/temp/train_flow_offline.py` with:

1. `--ema-decay 0.999`
2. `--eval-ema` or save EMA model inside checkpoint
3. `--freeze-visual-bn`
4. `--train-scope {all,flow_head,flow_head_aggregator}`
5. seeded demo selection and metadata logging

This keeps changes limited to the temporary diagnostic script while testing the strongest hypotheses.
