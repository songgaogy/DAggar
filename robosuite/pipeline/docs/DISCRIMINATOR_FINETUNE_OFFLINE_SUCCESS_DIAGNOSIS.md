# Discriminator Finetuning: Offline-Success Score Regression

## Scope

This document records the observed behavior and training dynamics of the run:

```text
outputs/discriminator-finetune/PickPlaceCereal_20260716_153919_run_cli
```

The investigation is limited to the training objective, optimization dynamics, and loss interactions. The selected training data recipe is treated as fixed. Post-training calibration design is outside the scope of this diagnosis.

## Run Configuration

The discriminator head was warm-started from the calibrated parent checkpoint and trained for 10 epochs, with 112 steps per epoch and 1,120 optimizer steps in total. The encoder remained frozen.

The objective was

$$
L
=
1.0L_{\mathrm{pre\text{-}nnPU}}
+0.025L_P
+0.0125L_N,
$$

where

$$
L_P
=
\mathbb{E}_{D_{\mathrm{off}}^P}
\left[\operatorname{softplus}(-g)\right],
$$

and

$$
L_N
=
\mathbb{E}_{D_{\mathrm{GT}}^N}
\left[\operatorname{softplus}(g)\right].
$$

Each optimizer step independently sampled:

- 256 pretrain-positive frames;
- 256 pretrain-unlabeled frames;
- 256 offline-positive frames;
- 256 offline GT-negative frames.

The active pools contained:

| Pool | Trajectories or windows | Frames |
| --- | ---: | ---: |
| Pretrain positive | 40 | 8,797 |
| Pretrain unlabeled | 50 | 20,000 |
| Offline positive | 36 | 5,368 |
| Offline GT-negative | 53 | 754 |
| Pretrain success calibration | 10 | 2,122 |

Offline-positive samples were not mixed into the pretrain-positive pool. They contributed only to the independent positive logistic risk $L_P$.

## Observed Training Dynamics

The total objective decreased substantially, but its components did not improve uniformly.

| Metric | Epoch 1 | Epoch 10 |
| --- | ---: | ---: |
| Total loss | 0.048617 | 0.013271 |
| Raw nnPU replay loss | 0.013822 | 0.000986 |
| Raw $L_P$ | 0.037478 | 0.215150 |
| Weighted $L_P$ | 0.000937 | 0.005379 |
| Raw $L_N$ | 2.708670 | 0.552503 |
| Weighted $L_N$ | 0.033858 | 0.006906 |
| nnPU clamp fraction | 0.982143 | 1.000000 |
| Offline-positive mean logit | 5.505747 | 4.850706 |
| GT-negative mean logit | 1.594145 | -0.856415 |
| Offline-positive minus GT-negative mean logit | 3.911602 | 5.707121 |

The discriminator therefore increased the mean separation between the two offline GT pools primarily by lowering GT-negative logits. At the same time, the mean offline-positive logit decreased and the positive logistic loss rose.

The epoch-level positive loss was not monotonic. It reached a maximum of 0.294749 at epoch 6. The interval-level positive loss also showed large spikes, with a maximum of 0.680981. This indicates substantial heterogeneity within the offline-positive pool rather than a uniform shift shared by every positive frame.

## Effective Gradient Imbalance

The loss signs are internally consistent:

$$
\frac{\partial L_P}{\partial g}
=
-\sigma(-g),
$$

so minimizing $L_P$ increases positive logits, while

$$
\frac{\partial L_N}{\partial g}
=
\sigma(g),
$$

so minimizing $L_N$ decreases negative logits.

No sign inversion was found in the implementation. The observed increase in $L_P$ arises within the joint objective rather than from an incorrect positive gradient.

At the beginning of training, offline-positive logits were already strongly positive. Consequently, $\sigma(-g)$ was small and the positive logistic gradient was saturated. GT-negative logits were initially positive, so $\sigma(g)$ was comparatively large. Although the configured positive weight was twice the negative weight, this did not imply a stronger effective positive gradient.

At epoch 1, weighted $L_N$ was approximately 36 times weighted $L_P$. In the first logged 10-step interval, weighted $L_N$ was approximately 313 times weighted $L_P$. The shared head could therefore reduce the total objective most rapidly by lowering offline-domain logits to fit the GT-negative pool, even when this increased the positive loss on part of the offline-success distribution.

## nnPU Replay Behavior

The nnPU negative-risk correction was clamped for 98.2% of batches in the first epoch and for effectively 100% of batches later in training. The estimated negative risk ended at -1.062126, while the used negative risk ended at zero.

Under this condition, the pretrain-unlabeled component contributed no gradient for most of training. The active nnPU gradient was dominated by its positive risk on the pretrain-positive pool. This behavior preserved strong scores on the pretrain success domain but did not prevent the offline-positive logits from decreasing under the supervised GT interaction.

## Operational Score Behavior

The discriminator uses

$$
\operatorname{failure\_score}=-g
$$

and predicts failure when

$$
-g\ge\tau.
$$

For this run, the final threshold was

$$
\tau=-5.495474,
$$

so a frame was considered safe only when its success logit exceeded 5.495474. The final offline-positive mean logit was 4.850706. Thus an offline-positive frame could have a strongly positive logit and a relatively small pointwise logistic loss while still falling on the operational failure side.

The threshold value is reported here only to explain how the saved model's raw offline-positive scores translate into its observed decisions. The calibration procedure itself is not treated as the cause under investigation.

## Existing Evaluation Evidence

The sampled offline-success visualization contained 10 successful trajectories with no intervention frames:

- total frames: 2,321;
- frames predicted as failure: 1,515;
- sampled offline-success false-alarm rate: 65.27%;
- 9 of the 10 trajectories were first flagged at frame 0.

In contrast, the existing success-rollout validation visualization remained substantially better. The failure is therefore concentrated in the offline-success score distribution rather than being a global inversion of the success label.

The offline GT-negative diagnostic reported:

- 754 GT-negative frames;
- 753 frames predicted as failure;
- 99.87% detection rate.

This is a train-set diagnostic, not held-out evidence. During 1,120 optimizer steps, each offline GT-negative frame was sampled approximately 380 times on average, compared with approximately 53 samples per offline-positive frame. Dataset size did not change the explicit risk coefficients, but the small negative pool was revisited much more frequently.

## Established Findings

The following points are supported by the run artifacts and implementation:

1. Offline success was kept separate from the pretrain-positive nnPU pool.
2. The positive and negative loss signs are correct.
3. The total loss decreased while $L_P$ increased.
4. Early optimization was dominated by the effective GT-negative gradient.
5. Positive logistic gradients were already saturated on much of the offline-positive pool.
6. The shared head improved offline GT separation mainly by lowering negative logits while also lowering the offline-positive mean logit.
7. nnPU replay was almost entirely clamped and supplied little or no pretrain-unlabeled gradient.
8. Strong performance on success-rollout data coexisted with a large offline-success false-alarm rate.
9. The ordinary positive logistic loss did not directly describe whether an offline-positive score lay on the saved model's operational safe side.

## Fixed Research Boundaries

For subsequent discussion of this issue:

- the offline-positive and GT-negative data selection is considered fixed;
- post-training calibration changes are not part of the current investigation;
- analysis should focus on the training loss, joint gradient interaction, optimizer dynamics, and the evolution of the discriminator head.
