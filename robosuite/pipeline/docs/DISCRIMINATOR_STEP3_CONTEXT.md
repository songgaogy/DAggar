# Discriminator Step 3 Context

## Purpose

This document provides self-contained context for Step 3 of the offline
discriminator update. It is written for readers who do not have access to the
repository. It records the current system, confirmed assumptions, observed
failure, and relevant run evidence. It intentionally does not propose solutions
or experimental changes.

## Project Context

The project combines a learned failure discriminator with the DIPOLE robotics
policy-training pipeline. The discriminator methodology used for pretraining is
considered fixed. Current work is focused on integrating and validating the
discriminator update inside the offline training pipeline.

The Step 3 task is to understand discriminator finetuning when a small amount of
ground-truth failure data is available. Step 3 is currently a brainstorming and
problem-definition stage; it does not require code modification, training, data
generation, or evaluation.

## Confirmed Step 3 Assumptions

The following facts have been confirmed by the user:

1. A rule-based procedure provides GT-negative frames near human intervention
   events. The temporal window size `N` is a hyperparameter.
2. The exact GT-negative rule is assumed to exist and is not part of the current
   discussion. The important property is that the resulting GT-negative dataset
   is small.
3. The frozen latent representation is already known to separate the relevant
   GT-positive and GT-negative examples. Representation separability is therefore
   not the current unknown.
4. The intended discriminator semantics are anticipatory risk prediction: the
   discriminator should identify risk before human intervention, rather than only
   recognize a physical failure after it has occurred.
5. The observed no-improvement result refers to the genuine warm-start run
   `PickPlaceCereal_20260715_172530`.
6. No improvement was judged from offline visualization: examples that the
   pretrained discriminator could not separate were still not separated after
   finetuning.

## Discriminator Semantics

The discriminator consists of a frozen visual-dynamics encoder followed by a
trainable MLP head.

- The encoder output used by the discriminator is the transformer layer-1
  latent.
- The encoder is frozen during offline discriminator finetuning.
- The head output `g(z)` is a success-likeness logit.
- The externally consumed failure score is `-g(z)`, so a larger score means more
  failure-like or risky.
- A frame is predicted as failure-like when its failure score is greater than or
  equal to a task-specific threshold.
- The threshold is calibrated using a held-out success-only calibration set. It
  is a percentile chosen from success failure-scores according to the configured
  false-positive budget `delta`.
- The canonical PickPlaceCereal checkpoint uses transformer layer 1 and action
  chunks. Its latent dimension is 14,272.

The latent is computed independently for each timestep. It is not a recurrent
trajectory classifier. When action chunks are enabled, the feature at time `t`
also contains a forward action chunk beginning at `t`.

## Original nnPU Pretraining

The pretrained head is learned with non-negative positive-unlabeled learning
(nnPU):

- Positive pool `P`: pre-completion frames from successful policy rollouts.
- Unlabeled pool `U`: complete failed rollouts, including both safe prefixes and
  failure portions.
- `pi_p`: the assumed fraction of success-like frames inside `U`.
- The canonical PickPlaceCereal action-chunk run uses `pi_p = 0.3` and the
  logistic surrogate.

The risk is:

```text
R_pu = pi_p * E_P[loss(+1, g)]
     + max(0, E_U[loss(-1, g)] - pi_p * E_P[loss(-1, g)])
```

The implementation uses the clamped form of the non-negative correction. When
the estimated negative-risk term is below zero, it is clamped to zero. The
implementation does not use the alternative gradient-ascent correction from the
canonical Kiryo nnPU algorithm.

## Offline Episode Data

The relevant PickPlaceCereal offline dataset contains:

| Quantity | Value |
| --- | ---: |
| Episodes | 50 |
| Transitions | 13,132 |
| Human-intervention transitions | 1,861 |
| Intervention ratio | 14.17% |
| Successful episodes | 41 |
| Manual-reset episodes | 9 |

Each collected transition contains observations, next observations, the action
executed in the environment, the policy-proposed action, a human-intervention
indicator, terminal information, and the discriminator score recorded during
collection.

During human intervention, the executed action is the human recovery action;
the policy action is stored separately. The Step 2 finetuning path deliberately
excludes intervention frames, human actions, and counterfactual policy actions.

The stored offline episode payload also has a field named `gt_fail`, but in the
current collector that field is a copy of the intervention indicator. The Step 3
GT-negative concept is the separately assumed rule-based label described above;
it should not be confused with the current stored field name.

## Current Step 2 Finetuning Data Routing

The existing finetuning pipeline splits every episode into maximal policy-only
segments at human-intervention boundaries.

Each policy segment is routed as follows:

- A segment enters `P` only when it is the final policy segment in the episode
  and its final policy-executed action directly completes the task successfully.
- Every other policy-only segment enters `U`. This includes segments ending at
  intervention, manual reset, maximum steps, environment termination, or an
  interrupted collection.
- Human-intervention frames are excluded entirely.
- The current finetuning route does not read or optimize against any explicit
  GT-negative pool.

The policy-only segments are encoded independently. Action chunks are padded
within each segment and do not cross intervention or episode boundaries.

## Pool Composition in the Warm-Start Run

The `PickPlaceCereal_20260715_172530` run combined the extracted pretraining
pools with the newly encoded offline policy segments:

| Source and pool | Trajectories or segments | Frames |
| --- | ---: | ---: |
| Pretrain `P` | 40 trajectories | 8,797 |
| Pretrain `U` | 50 trajectories | 20,000 |
| Pretrain success calibration | 10 trajectories | 2,122 |
| Offline `P` | 36 segments | 5,368 |
| Offline `U` | 58 segments | 5,903 |
| Excluded human-intervention data | — | 1,861 |

The combined training pools therefore contained:

- 14,165 positive frames;
- 25,903 unlabeled frames;
- 2,122 held-out success-calibration frames.

Training batches used an exact 50/50 `P`/`U` composition. Within each pool,
frames were sampled uniformly with replacement. Pretrain and offline frames were
not assigned separate class priors or separate nnPU risks. The parent
checkpoint's `pi_p = 0.3` was used for the combined pool.

## Warm-Start Run Identity

`PickPlaceCereal_20260715_172530` is the relevant genuine warm-start run:

- Parent head: the pretrained PickPlaceCereal action-chunk nnPU checkpoint.
- Encoder: frozen transformer layer-1 dynamics encoder.
- Optimizer: AdamW.
- Learning rate: `3e-5`.
- Weight decay: `1e-4`.
- Batch size: 512.
- Epochs: 50.
- Loss surrogate: logistic.
- nnPU class prior: `pi_p = 0.3`.
- Non-negative correction: enabled with `beta = 0`.
- Calibration: original held-out pretrain success calibration only.

Two other local runs, `PickPlaceCereal_20260715_175113` and
`PickPlaceCereal_20260715_191438`, initialized replacement heads rather than
warm-starting the pretrained head. They are not the runs referenced by the
confirmed no-improvement observation.

## Observed Optimization Behavior

TensorBoard scalars from `PickPlaceCereal_20260715_172530` show:

| Measurement | First epoch | Final epoch |
| --- | ---: | ---: |
| Mean nnPU risk | 0.0270 | 0.000872 |
| Mean positive risk | 0.0179 | 0.000872 |
| Mean estimated negative risk | -0.534 | -1.110 |
| Mean negative risk used by the loss | 0.00912 | 0.0 |
| Batches triggering correction | 141 / 156 | 156 / 156 |
| Correction fraction | 90.38% | 100% |

Because the implementation clamps negative risk below zero, a clamped batch has
zero gradient through the unlabeled negative-risk term. By the final epoch, all
batches were clamped, and the reported total risk was numerically equal to the
positive risk. Thus the late-stage optimization signal came from the positive
term rather than an explicit GT-negative objective.

The training loss decreased substantially, but the confirmed offline
visualization result showed no new separation on the previously unresolved
offline examples.

## Distribution of the Offline Unlabeled Pool

The original nnPU unlabeled pool consists of complete failed rollouts collected
without using the current offline routing rule. The offline `U` pool has different
provenance:

- it contains policy segments selected around intervention and other episode
  boundaries;
- it was collected with a discriminator active in the collection process;
- it contains hard or ambiguous examples for that collection setup;
- a complete segment may contain a long safe prefix and only a small region near
  an intervention event;
- it is merged with the original pretrain `U` and trained with the same inherited
  class prior.

In the genuine warm-start run, offline `U` contributed 5,903 of 25,903 combined
unlabeled frames, or approximately 22.8%.

The offline dataset was collected using an older non-action-chunk nnPU
checkpoint, while the warm-start parent and extracted latent manifest correspond
to the newer action-chunk checkpoint. The current finetuning contract records
this provenance but does not require the collection-time discriminator checkpoint
to equal the warm-start parent checkpoint.

## Current Visualization Semantics

The standalone finetuned visualizer produces videos and score PDFs for benchmark
trajectories and collected offline episodes. It does not run a quantitative
benchmark evaluation.

For collected offline episodes:

- the visualizer does not load a rule-based GT-negative mask;
- the current stored `gt_fail` field is not used as the visualization GT mask;
- discriminator annotations are hidden on intervention frames;
- trajectory scoring uses the recorded episode action stream, which includes
  human-executed actions during intervention;
- no offline GT-negative AUROC, AUPRC, event recall, false-positive rate, or lead
  time is reported.

The confirmed Step 3 observation is therefore a qualitative score-separation
observation from the offline visualization output.

## Fixed Constraints and Scope

- The pretrained discriminator methodology is fixed unless explicitly reopened.
- The dynamics encoder remains frozen.
- Transformer layer-1 latent semantics and checkpoint provenance are preserved.
- Tensor computation must use CUDA; CPU tensor fallback is not allowed.
- TensorBoard is the experiment logger.
- Step 3 currently assumes scarce rule-based GT-negative data is available.
- Step 3 currently assumes the frozen latent is separable.
- The active task and referenced run are PickPlaceCereal and
  `PickPlaceCereal_20260715_172530`.
- This document describes the situation only and contains no proposed
  improvement method.
