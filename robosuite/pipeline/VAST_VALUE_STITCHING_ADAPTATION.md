# VAST Value-Stitching Adaptation for Offline DIPOLE

## Status

This document describes the active value-learning method in the DIPOLE pipeline.
The implementation is based on `docs/vast_source-code/agents/vast.py`, but it is
an explicit adaptation rather than the complete upstream VAST agent.

## Fixed design

- Keep the DIPOLE dual-branch policy, routed-sigmoid weights, and frozen nnPU
  discriminator.
- Do not introduce twin Q, `pi_stitch`, `pi_exec`, or rejection sampling.
- Share and freeze the existing `SharedDynamicsEncoder`.
- Treat `k` as macro chunks: one macro chunk contains `H` environment steps.
- Default to `single_vast`; retain `ensemble_lcb` as a configured ablation.

## Joint G/V update

For a valid start `t`, sample `k` uniformly from the legal future macro chunks.
For `k >= 2`, sample `j` uniformly from `[1, k-1]`. Sampling never crosses an
episode boundary or terminal.

`G(s_t, s_{t+kH}, k)` receives two losses:

1. Monte-Carlo regression to the discounted k-chunk return.
2. Composition consistency:

```text
G(t, k) = G(t, j) + gamma^(jH) G(t+jH, k-j)
```

Both sides participate in gradient computation. The composition coefficient is
`0.5`. Samples with `k < 2` follow the source mask and do not update G.

V uses expectile regression with default expectile `0.9` and target

```text
stopgrad(G(s_t, s_{t+kH}, k)
         + gamma^(kH) * (1 - done) * V_target(s_{t+kH}))
```

G is updated only by Monte-Carlo and composition losses. Warmup and offline
finetuning use the same joint update semantics.

## Offline DIPOLE integration

Phase A finetunes G/V on collected `policy_bc` plus warmup transitions. Phase B
samples one legal `k` per policy window with the configured seed and computes

```text
A_stitch = G(s, s_k, k) + gamma^(kH) V_target(s_k) - V(s)
```

The existing routed-sigmoid policy uses this advantage directly. If fewer than
two future macro chunks remain, the window falls back to the existing TD1
advantage. GAE remains a diagnostic only.

## Configuration and checkpoints

The canonical Hydra namespace is `algorithm.vast`. Relevant settings include V
mode, maximum K, composition coefficient, sampling seed, expectile, G/V learning
rates, reward coefficients, batch size, and joint steps.

New warmup and finetuned checkpoints use schema v7 and save G, V, target V,
optimizers, V mode, K, macro horizon, reward/encoder metadata, and sampling seed.
New artifact names are `vast_state.pt`, `vast_state_finetuned.pt`, and
`vast_offline_transitions.pt`.

Existing schema-v6 artifacts with legacy naming may be loaded only through the
explicit read-compatibility path. Schema-v5 and older value-only checkpoints are
not valid VAST initialization sources.

## Visualization and logging

Diagnostics include sampled k, future-frame index, G, bootstrap value, stitched
target/advantage, Monte-Carlo error, composition residual, and TD1 fallback. The
visualizer shows current and selected future frames side by side and retains the
existing V, TD1, GAE, and nnPU diagnostics.

TensorBoard and `run_info.json` identify the method as
`vast_value_stitching_adaptation`, not complete upstream VAST.
