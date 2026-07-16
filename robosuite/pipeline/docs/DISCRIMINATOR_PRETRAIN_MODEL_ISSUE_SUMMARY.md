# Discriminator Pretrain Model Issue Summary

## Scope

This document summarizes the observed issues of the current pretrained
PickPlaceCereal nnPU discriminator as a starting point for frozen-encoder,
head-only offline finetuning. It records model behavior, score geometry,
optimization evidence, and downstream consequences. It does not specify or
recommend a corrective method.

The parent checkpoint is:

```text
checkpoints/dyn_disc/pu_bce_eval_robosuite-chunk/
run_20260715_121551_PickPlaceCereal/checkpoints/pu_bce_head.pth
```

The head output is the raw success-likeness logit `g`. The operational failure
score is `-g`, and failure is predicted when `-g >= tau`.

## Parent Score Geometry

The parent checkpoint uses success-only percentile calibration with
`delta=10`. Its PickPlaceCereal calibration set contains 2,122 frames.

| Parent calibration statistic | Failure score `-g` | Success logit `g` |
| --- | ---: | ---: |
| Mean | -11.742556 | 11.742556 |
| Standard deviation | 5.580984 | 5.580984 |
| Minimum / maximum | -18.188507 / -1.319653 | 1.319653 / 18.188507 |
| Operational threshold / boundary | `tau=-4.201930` | `g=4.201930` |

Consequently, zero is not the operational center of the pretrained score. A
frame is on the parent's safe side only when `g > 4.201930`, while the mean
calibration success logit is approximately `11.74`.

The checkpoint stores the raw head output and a task-specific threshold. It
does not define a normalized logit coordinate in which zero is the calibrated
decision boundary or in which one score unit has a fixed distributional
meaning.

## Positive-Loss Saturation

The positive logistic risk is

$$
L_P = \mathbb{E}[\operatorname{softplus}(-g)],
$$

with derivative

$$
\frac{\partial L_P}{\partial g} = -\sigma(-g).
$$

At the parent calibration mean `g=11.742556`, the pointwise gradient magnitude
is approximately `8e-6`. Positive samples in this score region therefore have
almost no gradient under the ordinary positive logistic loss. This is an
optimization saturation effect: `torch.nn.functional.softplus` remains
numerically stable, but the intended positive gradient becomes very small.

The previous three-risk finetuning run showed the downstream effect directly.
Offline-positive samples began with strongly positive logits, while the
GT-negative samples still produced comparatively large negative-class
gradients. The epoch-1 weighted GT-negative loss was approximately 36 times the
weighted offline-positive loss; in the first logged 10-step interval, the ratio
was approximately 313. The shared head could lower the total objective mainly
by reducing GT-negative logits even when part of the offline-positive score
distribution also moved downward.

## nnPU Replay During Finetuning

The pretrained positive and unlabeled pools are replayed during offline
finetuning with the parent nnPU semantics. In the previous three-risk run, the
nnPU negative-risk correction was clamped for 98.2% of epoch-1 batches and for
effectively 100% of later batches. In the positive safety-margin run, the clamp
fraction was again 98.2% in epoch 1 and 100% in epoch 10.

When clamped, the used nnPU negative risk is zero. The pretrained unlabeled
replay branch therefore supplies no gradient for most finetuning steps. The
remaining nnPU replay gradient is dominated by the pretrained positive risk.
As a result, replaying the parent data does not preserve both sides of the
parent score geometry during head-only adaptation.

This clamp evidence describes the parent checkpoint's behavior when replayed
inside the current finetuning objectives. It does not establish that every
batch was clamped throughout the original pretraining run.

## Calibration-Boundary Drift

All compared checkpoints use the same score convention and the same
success-only `delta=10` calibration rule, but the raw-logit decision boundary
changes substantially after head finetuning.

| Checkpoint | Failure threshold `tau` | Equivalent safe-logit boundary `-tau` |
| --- | ---: | ---: |
| Parent pretrain checkpoint | -4.201930 | 4.201930 |
| Previous three-risk finetune | -5.495474 | 5.495474 |
| Positive safety-margin finetune | -7.776727 | 7.776727 |

The calibration rule itself remains at the same percentile. The numerical
boundary changes because the shared head changes the location and scale of the
pretrain calibration logits. The raw value of `g` therefore does not retain a
fixed operational meaning across head updates.

This effect also separates the loss target from the final detector boundary.
In the positive safety-margin run, the fixed training target was
`g=5.201930`, derived from the parent boundary, while post-training calibration
placed the operational boundary at `g=7.776727`.

## Domain-Dependent Downstream Behavior

The raw score distribution does not move uniformly across the pretrain
calibration domain and the offline-success domain.

In the previous three-risk run:

- the offline-positive mean logit decreased from `5.505747` in epoch 1 to
  `4.850706` in epoch 10;
- the final safe-logit boundary was `5.495474`;
- 1,515 of 2,321 sampled offline-success frames were predicted as failure;
- the sampled offline-success false-alarm rate was 65.27%.

In the positive safety-margin run:

- the final offline-positive mean logit was `10.449436`;
- the final safe-logit boundary was `7.776727`;
- 740 of the same 2,321 sampled offline-success frames were predicted as
  failure;
- the sampled offline-success false-alarm rate was 31.88%.

The higher offline-positive mean does not eliminate false alarms because the
distribution is heterogeneous and its lower-scoring tail remains below the
recalibrated boundary. The mean score alone therefore does not describe the
operational behavior of the full offline-success distribution.

The GT-negative train-set diagnostic changed from `753/754` detected frames
(99.87%) in the previous run to `746/754` (98.94%) in the safety-margin run.
These values are training-pool diagnostics rather than held-out estimates.

## Established Issue Summary

The current evidence supports the following statements about the pretrained
model as a finetuning starting point:

1. The parent success logits occupy a large positive range, and zero is not the
   calibrated decision boundary.
2. The raw score has no fixed normalized origin or scale across head updates.
3. Ordinary positive logistic gradients are strongly saturated over much of
   the parent success score range.
4. The pretrained unlabeled nnPU replay gradient is inactive for most observed
   finetuning batches because the negative risk is clamped.
5. Updating the shared head shifts the calibration and offline domains by
   different amounts.
6. Post-training percentile calibration can move the operational raw-logit
   boundary substantially even when the percentile rule and calibration data
   recipe are unchanged.
7. A high mean success logit can coexist with a substantial operational
   false-alarm rate because lower-tail frames remain below the calibrated
   boundary.

## Findings Not Established by the Current Evidence

The current evidence does not show:

- a numerical instability or overflow in `softplus`;
- a sign inversion in the positive or negative logistic losses;
- a global inversion of success and failure labels;
- that the frozen dynamics encoder alone is the root cause;
- that the original pretraining run had the same nnPU clamp fraction at every
  stage as the later finetuning runs.

## Recorded Evidence

- `robosuite/pipeline/docs/DISCRIMINATOR_FINETUNE_OFFLINE_SUCCESS_DIAGNOSIS.md`
- `outputs/discriminator-finetune/PickPlaceCereal_20260716_153919_run_cli/`
- `/data/gy/robosuite/outputs/discriminator-finetune/`
  `PickPlaceCereal_20260716_165327_positive_safety_margin_mu1_d1_t1_seed0/`
- `robosuite/discriminator/dyn_disc/detectors/pu_bce.py`
- `robosuite/pipeline/offline/discriminator/objectives.py`
- `robosuite/pipeline/offline/discriminator/trainer.py`
