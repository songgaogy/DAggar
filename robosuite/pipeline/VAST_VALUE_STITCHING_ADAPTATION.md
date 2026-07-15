# VAST Value-Stitching Adaptation for Offline DIPOLE

## Status

This document describes the active value-learning method in the DIPOLE pipeline. The implementation is based on `docs/vast_source-code/agents/vast.py`, but it is an explicit adaptation rather than the complete upstream VAST agent.

## Fixed design

- Keep the DIPOLE dual-branch policy, routed-sigmoid weights, and frozen nnPU discriminator.
- Do not introduce twin Q, `pi_stitch`, `pi_exec`, or rejection sampling.
- Share and freeze the existing `SharedDynamicsEncoder`.
- Treat `k` as macro chunks: one macro chunk contains `H` environment steps.
- Default to `single_vast`; use `indep_ensemble` for the independent V ablation.

## Joint G/V update

For a valid start `t`, sample `k` uniformly from the legal future macro chunks. For `k >= 2`, sample `j` uniformly from `[1, k-1]`. Sampling never crosses an episode boundary or terminal.

`G(s_t, s_{t+kH}, k)` receives two losses:

1. Monte-Carlo regression to the discounted k-chunk return.
2. Composition consistency:

```text
G(t, k) = G(t, j) + gamma^(jH) G(t+jH, k-j)
```

Both sides participate in gradient computation. The composition coefficient is `0.5`. Samples with `k < 2` follow the source mask and do not update G.

V uses expectile regression with default expectile `0.9` and target

```text
stopgrad(G(s_t, s_{t+kH}, k)
         + gamma^(kH) * (1 - done) * V_target(s_{t+kH}))
```

G is updated only by Monte-Carlo and composition losses. Warmup and offline finetuning use the same joint update semantics.

## Offline DIPOLE integration

Phase A finetunes G/V on collected `policy_bc` plus warmup transitions. Phase B uses the finetuned VAST V / target-V pair to compute

```text
delta_t = r_chunk(t) + gamma^H * m_t * V_target(s_{t+H}) - V(s_t)
A_t     = delta_t + gamma^H * lambda * m_t * A_{t+H}
```

Here `m_t=1-done_t`; the recursive future term is zero when no valid same-section `t+H` successor row exists. The GAE recursion advances by one macro chunk and uses `lambda=0.6` by default. The existing routed-sigmoid policy uses `A_t` directly. `td1` is an explicit ablation; the stitched advantage is retained only for Phase-A VAST learning and diagnostics.

## Configuration and checkpoints

The canonical Hydra namespace is `algorithm.vast`. Relevant settings include V mode, maximum K, composition coefficient, sampling seed, expectile, G/V learning rates, reward coefficients, batch size, and joint steps.

New warmup and finetuned checkpoints use schema v7 (`single_vast`) or schema v8 (`indep_ensemble`) and save G, V, target V, optimizers, V mode, K, macro horizon, reward/encoder metadata, and sampling seed. New artifact names are `vast_state.pt`, `vast_state_finetuned.pt`, and `vast_offline_transitions.pt`.

Existing schema-v6 artifacts with legacy naming may be loaded only through the explicit read-compatibility path. Schema-v5 and older value-only checkpoints are not valid VAST initialization sources.

## Visualization and logging

Diagnostics include sampled k, future-frame index, G, bootstrap value, stitched target/advantage, Monte-Carlo error, and composition residual. The visualizer shows current and selected future frames side by side and retains the existing V, TD1, GAE, and nnPU diagnostics. Stitched quantities diagnose Phase-A VAST but do not provide Phase-B policy weights.

TensorBoard and `run_info.json` identify the method as `vast_value_stitching_adaptation`, not complete upstream VAST.
