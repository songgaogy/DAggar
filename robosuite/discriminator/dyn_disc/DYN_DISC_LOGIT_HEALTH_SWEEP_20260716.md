# dyn_disc Logit Health Sweep (2026-07-16)

## Outcome

The approved 22-trial sweep completed without CUDA, OOM, non-finite, data-overlap, checkpoint, manifest, or TensorBoard failures. The final strict five-task selection is:

- config: `cap_c5_l1e2`
- common epoch: `1`
- soft-cap: `C=5`, `lambda=1e-2`, `T=1`
- threshold normalization: disabled
- macro task score: `0.755201`
- worst task score: `0.652630` (`PickPlaceMilk`)
- global score: `0.703915`
- maximum saturation fraction: `0.008114`

This is the only config/epoch pair that is healthy on all five tasks. Later epochs often have higher evaluation scores but are excluded by the approved logit-health gate.

The machine-readable bundle, including all metrics, checkpoint paths, and SHA-256 hashes, is at:

```text
checkpoints/dyn_disc/pu_bce_eval_robosuite-chunk/
trick_sweep_20260716_v1/best_bundle.json
```

## Implemented Behavior

The nnPU label and risk semantics are unchanged. The implementation adds:

- a persistent `BCEHead.logit_center`, with effective logit `g = h - b`;
- detached epoch-boundary success-calibration normalization;
- the effective-logit soft-hinge penalty `lambda * mean(T * softplus((abs(g) - C) / T))`;
- CUDA-computed raw/effective logit diagnostics for positive, unlabeled, and success-calibration pools;
- epoch `1, 2, 5, 10, 20` snapshots and post-training cached evaluation;
- eval-success pre-done frame false-alarm rate, specificity, and trajectory alarm rate;
- TensorBoard training, health, and benchmark metrics;
- strict health and deterministic stage-one/final selection;
- CUDA hard failure instead of silent requested-CUDA fallback;
- checkpoint provenance, split IDs, config, epoch, training history, model and normalizer hashes, and legacy `logit_center=0` loading.

## Protocol

All trials use seed `0`, 20 epochs, AdamW, learning rate `3e-4`, weight decay `1e-4`, batch size `512`, head `512 x 3`, logistic nnPU, `pi_p=0.3`, `beta=0`, `delta=10`, calibration fraction `0.2`, transformer layer `1`, and action chunks. Each task uses at most 50 training success trajectories, 50 whole failure trajectories, 50 eval success trajectories, and 50 eval failure trajectories. The available eval failure counts are 45 Cereal, 48 Round, 47 Square, 34 Milk, and 46 Stack.

`H` below means the checkpoint passes all approved health constraints; `X` means it fails at least one constraint. Scores are still shown for unhealthy checkpoints, but unhealthy checkpoints cannot be selected.

The health gate requires every effective-logit pool to be finite, have `abs_p99 <= 9.21`, and have `abs(g) > 9.21` fraction at most `0.05`. Normalized trials additionally require `abs(tau) <= 0.1`.

## Stage One: PickPlaceCereal

| Config | e1 | e2 | e5 | e10 | e20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | H 0.794344 | H 0.791716 | X 0.799212 | X 0.798158 | X 0.794947 |
| cap_c5_l1e2 | H 0.796152 | H 0.803264 | X 0.796835 | H 0.801849 | H 0.797255 |
| cap_c5_l1e3 | H 0.792989 | H 0.793272 | X 0.791206 | X 0.797982 | X 0.796681 |
| cap_c8_l1e2 | H 0.794339 | H 0.788930 | X 0.791473 | X 0.794082 | H 0.798820 |
| cap_c8_l1e3 | H 0.793981 | H 0.794005 | X 0.796382 | X 0.795157 | X 0.796088 |
| norm_cap_c5_l1e2 | H 0.796152 | H 0.795011 | H 0.805269 | H 0.797926 | H 0.789135 |
| norm_cap_c5_l1e3 | H 0.792989 | H 0.785239 | H 0.797656 | X 0.802619 | X 0.790451 |
| norm_cap_c8_l1e2 | H 0.794339 | X 0.792882 | H 0.799058 | X 0.796669 | X 0.790511 |
| norm_cap_c8_l1e3 | H 0.793981 | X 0.794340 | X 0.785738 | X 0.798232 | X 0.795168 |
| norm_only | H 0.794344 | X 0.795507 | X 0.797411 | H 0.797465 | X 0.792392 |

The stage-one Top-2 are:

1. `norm_cap_c5_l1e2`, epoch 5, score `0.805269`, maximum effective `abs_p99=7.071`, saturation `0`, and `abs(tau)=1.25e-7`.
2. `cap_c5_l1e2`, epoch 2, score `0.803264`, maximum effective `abs_p99=8.796`, and saturation `0`.

The first candidate improves the historical Cereal reference from trajectory AUROC `0.9893` to `0.9951`, frame AUROC `0.9221` to `0.9332`, and frame F1 `0.8819` to `0.8962`.

## Stage Two

### NutAssemblyRound

| Config | e1 | e2 | e5 | e10 | e20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | H 0.787614 | X 0.782058 | X 0.815668 | X 0.770897 | X 0.780848 |
| cap_c5_l1e2 | H 0.795139 | X 0.780780 | X 0.786862 | X 0.792166 | X 0.781170 |
| norm_cap_c5_l1e2 | H 0.795139 | X 0.816006 | X 0.797804 | H 0.787320 | X 0.757611 |

### NutAssemblySquare

| Config | e1 | e2 | e5 | e10 | e20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | H 0.748465 | X 0.766602 | X 0.777363 | X 0.794427 | X 0.785061 |
| cap_c5_l1e2 | H 0.749649 | X 0.763743 | X 0.782265 | X 0.791718 | H 0.799020 |
| norm_cap_c5_l1e2 | H 0.749649 | X 0.755075 | X 0.780822 | H 0.806940 | H 0.787823 |

### PickPlaceMilk

| Config | e1 | e2 | e5 | e10 | e20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | X 0.637951 | X 0.658290 | X 0.629513 | X 0.677395 | X 0.677993 |
| cap_c5_l1e2 | H 0.652630 | X 0.653356 | X 0.666429 | X 0.690329 | X 0.723214 |
| norm_cap_c5_l1e2 | X 0.652630 | X 0.663908 | X 0.674060 | X 0.710858 | X 0.716577 |

### Stack

| Config | e1 | e2 | e5 | e10 | e20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | H 0.779409 | X 0.780135 | X 0.797371 | X 0.785575 | X 0.776512 |
| cap_c5_l1e2 | H 0.782434 | X 0.779327 | X 0.781937 | X 0.791233 | X 0.783473 |
| norm_cap_c5_l1e2 | H 0.782434 | H 0.778247 | H 0.793707 | H 0.797115 | X 0.791335 |

## Final Bundle Metrics

| Task | Task score | Traj AUROC | Frame AUROC | Frame F1 | Event F1 | Success specificity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PickPlaceCereal | 0.796152 | 0.992444 | 0.898291 | 0.888198 | 0.337662 | 0.864167 |
| NutAssemblyRound | 0.795139 | 0.937500 | 0.902796 | 0.896290 | 0.351333 | 0.887776 |
| NutAssemblySquare | 0.749649 | 0.911064 | 0.844029 | 0.795751 | 0.297972 | 0.899427 |
| PickPlaceMilk | 0.652630 | 0.869412 | 0.621243 | 0.635821 | 0.309867 | 0.826806 |
| Stack | 0.782434 | 0.934348 | 0.910598 | 0.910979 | 0.360465 | 0.795780 |

| Task | P abs-p99 | U abs-p99 | Calib abs-p99 | Max saturation | Success frame FAR | Success trajectory alarm rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PickPlaceCereal | 7.365 | 7.079 | 7.058 | 0.000000 | 0.135833 | 0.68 |
| NutAssemblyRound | 8.577 | 8.390 | 8.578 | 0.000000 | 0.112224 | 0.48 |
| NutAssemblySquare | 7.551 | 4.895 | 6.190 | 0.000000 | 0.100573 | 0.40 |
| PickPlaceMilk | 9.205 | 9.134 | 9.191 | 0.008114 | 0.173194 | 0.78 |
| Stack | 6.986 | 2.873 | 6.785 | 0.000000 | 0.204220 | 0.64 |

The uniform epoch constraint changes the Cereal trade-off: the final bundle's Cereal model has higher trajectory AUROC and frame F1 than the historical reference, but lower frame AUROC (`0.8983` versus `0.9221`). The Cereal-only stage-one best is therefore not interchangeable with the final uniform bundle.

## Diagnostics and Limitations

- `PickPlaceMilk` is the binding task. Its selected P-pool `abs_p99=9.205` is close to the `9.21` limit, although its maximum saturation fraction remains only `0.008114`. This checkpoint is healthy by the approved gate but has little p99 margin.
- The cap-only final bundle has `logit_center=0`, so raw and effective logits are identical; its healthy output is not produced by hiding a raw offset.
- Normalization reliably makes calibrated `tau` numerically zero, but it does not prevent raw bias drift. Across normalized trials, the largest absolute center is `31.662` (Stack epoch 20), and some checkpoints remain unhealthy after centering. The raw/effective diagnostics preserve this evidence.
- Several unhealthy later checkpoints score better than the final epoch-1 bundle. For example, the cap-only epoch-20 global score would be `0.750020`, but it fails the health gate on multiple tasks. It is intentionally excluded.
- Event F1 and success trajectory alarm rates remain weak. The selected model improves logit health without resolving all temporal false alarms.
- This is a seed-0 exploratory selection on the benchmark eval set. The same eval set is used for checkpoint/config selection, so the reported metrics are selection estimates rather than an untouched final test estimate.
- Model checkpoints contain train-success, success-calibration, and unlabeled failure IDs. Eval-failure IDs are in each manifest, while all eval success and failure IDs are retained in the bundle's per-trajectory benchmark data rather than duplicated inside the model checkpoint.

## Verification and Artifacts

- CUDA unit/integration tests: `33 passed`.
- CUDA end-to-end smoke: train, epoch snapshot, eval, TensorBoard, manifest.
- Python compile, CLI help, shell syntax, and `git diff --check`: passed.
- Formal outputs: 22 trial summaries, 110 epoch checkpoints, 110 epoch benchmark JSON files, 22 TensorBoard event files, and 22 manifests.
- Data disjointness assertions passed for every formal trial.

No nnPU label semantics, failure-timing restrictions, success-percentile calibration rule, split disjointness, or unrelated dependency versions were changed.
