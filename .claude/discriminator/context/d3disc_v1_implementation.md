# D3-Disc v1 — Implementation Summary

> Conversation: implemented D3-Disc end-to-end (`robosuite/discriminator/d3disc/`), then retrofitted a latent-dynamics predictor so the discriminator feature is LPB-compatible instead of raw flow_multi `z_t`.
> Source design: `.claude/discriminator/conversation_context.md`, `.claude/discriminator/sfv_discriminator_design.md`, `.claude/discriminator/implementation_plan.md`.

---

## 1. What was built

### 1.1 Package layout (new, no `lpb/` modifications)

```
robosuite/discriminator/d3disc/
├── __init__.py
├── encoder.py           # FlowMultiEncoderWrapper (frozen flow_multi, cache-aware)
├── filter.py            # knn_sqdist, compute_f3_weights, weighted_knn_sqdist
├── detector.py          # D3Detector: fit_banks -> calibrate -> score
├── dataset.py           # LatentFlowDynamicsDataset (transitions from cached z_t + HDF5 s,a)
├── trainer.py           # D3DynamicsTrainer — trains predictor on cached latents
├── train_dynamics.py    # CLI entry for dynamics training
├── dynamics_feature.py  # D3FeatureExtractor — inference-time 3-concat feature
├── d3_benchmark.py      # D3BenchmarkDiscriminator (benchmark adapter)
├── visualize.py         # D3Visualizer (mp4 + pdf)
├── README.md
└── scripts/
    ├── train_d3_dynamics.sh
    ├── run_d3_benchmark.sh
    ├── visualize_d3.sh
    └── ablate_omega_beta_k.sh

data/utils/benchmark/examples/
└── run_d3.py            # CLI driver (parallel to run_lpb.py)
```

### 1.2 Two feature modes (controlled by `DYN_CKPT`)

- **Raw `z_t`** (DYN_CKPT empty): 256-D flow_multi `task_scene_cond` per frame. Fast, no training. No action/dynamics regularization.
- **LPB-style projected** (DYN_CKPT set): `phi = [obs_proj(z), proprio_proj(s), mean(action_proj(a_chunk))]`, 1536-D by default (3 * d_model=512). L2-normalized (like LPB). Requires a trained `DynamicsPredictor`.

The `DynamicsPredictor` architecture is reused verbatim from `lpb.model.DynamicsPredictor`; only `latent_dim` switches from 512 (ResNet-18) to 256 (flow_multi aggregator output).

### 1.3 Score

```
lambda_t = (1+omega) * d^2(phi_t, D+) - omega * d^2_weighted(phi_t, D-)
```

- Pos bank `D+` from success trajectories (shared across tasks per D8).
- Neg bank `D-` from fail trajectories, F3 soft-cleaned:
  `w_j^- = sigmoid(beta*(d^2_knn(phi_j, D+) - kappa))`, with auto `beta = 1/MAD`, `kappa = median` of pos self-kNN.
- `weighted_knn_sqdist` folds log-weights as `d^2_eff = ||q-b||^2 - 2*sigma^2*log(w)`.
- Per-task `tau` via percentile on aggregated lambdas over held-out success calibration frames.

### 1.4 Data / cache reuse

- All 650-demo-per-task `.npz` cache under `data/.lpb_score_cache/<task>/<sha1>.npz` is reused **byte-exactly** on cache HIT. `cache_key` byte-matches the historical git-5056929 `lpb_score.core.policy_encoder.FlowMultitaskEncoder.cache_key` formula.
- Cache contains `latents (T,256) float32`, `actions (T,7) float32`, `task_name`, `file_path` (source HDF5 absolute path), `demo_key`. **No proprio in cache** — trainer reads states fresh from HDF5.
- Cache coverage: expert + success_rollout + fail_rollout all present for 5/6 tasks; PandaLift has no fail_rollout; PickPlaceCan has sparse/older cache entries that may not match `data/` layout.
- `BenchmarkTrajectory.source_hdf5_path` + `source_demo_key` map to cache keys cleanly.

---

## 2. Verified smoke-tests

### Phase A — cache bit-exactness
- cache_key sha1 match: 6/6 tasks ✓
- `load_or_encode_demo` on existing cache: `max|diff| = 0.0` ✓
- Fresh re-encode (`encode_fresh`) vs cached latents:
  - 5/6 tasks: `max|diff| <= 5e-6`
  - PandaLift: `max|diff| = 1.2e-4` (cuDNN kernel-selection jitter across machines; KNN-level impact negligible)
- Self-consistency of the new encoder: `max|diff| = 0.0`
- **Important**: `encoder_batch_size=256` is the default that minimizes drift vs the historical cache (96/128 give ~5e-4, 256/512 give ~1e-6). Never lower without re-verifying.

### Phase C — benchmark end-to-end (PickPlaceBread, 50 traj)

| config | feature | traj AUROC | traj AUPRC | F1 |
|---|---|---|---|---|
| omega=0, raw z | 256-D z_t | 0.9243 | 0.8562 | 0.902 |
| omega=0.5, raw z | 256-D z_t | 0.9758 | 0.9690 | 0.936 |
| omega=0.5, DYN_CKPT (2-epoch smoke) | 1536-D phi | 0.9823 | 0.9808 | 0.920 |

Multi-task omega=0.5 on 4 tasks (200 traj): AUROC 0.993, AUPRC 0.992, F1 0.957.

### Dynamics trainer (2-epoch smoke, 10 traj/kind, PickPlaceBread)
- Total 6298 samples; latent_dim=256, proprio_dim=71, action_dim=7.
- Loss: 2.45 -> 0.04 in 2 epochs; `latent_mse` 1.35 -> 0.026.
- Checkpoint saves to `checkpoints/d3disc/dynamics/<RUN_NAME>/d3_dynamics.pt` with full payload (state_dict + args + dim metadata + `policy_ckpt_path` + `proprio_indices`).

---

## 3. Locked decisions from this session

| ID | Choice | Notes |
|---|---|---|
| **encoder-batch-size** | default `256` | Minimizes cuDNN drift vs historical cache (verified). |
| **driver location** | `data/utils/benchmark/examples/run_d3.py` | Matches repo convention over implementation_plan.md §5.1 literal path; scripts invoke `-m data.utils.benchmark.examples.run_d3`. |
| **fit+eval split** | single script | `run_d3_benchmark.sh` does both (D3-Disc has no gradient training apart from the optional dynamics predictor). |
| **feature mode default** | `DYN_CKPT` empty -> raw z_t | Projected phi only when explicitly provided. |
| **projected feature dim** | 3 × d_model = 1536, L2-normalized | LPB-compatible geometry. |
| **dynamics training** | predictor only, latents from cache | Encoder stays frozen. Training is latent->latent, no image forward per batch. |
| **training data sources** | expert + success_rollout | Fail excluded from dynamics training (match LPB). |
| **horizon** | 1 | Step-wise. D2 locked K=0; K-step rollout deferred. |
| **F3** | static only | D11. EM iteration deferred. |
| **Share banks across tasks** | `SHARE_BANKS=1` default | D8. Per-task tau. `SHARE_BANKS=0` available as ablation. |

---

## 4. User-facing knobs

### `run_d3_benchmark.sh`

| Env var | Default | Meaning |
|---|---|---|
| `POLICY_CKPT` | `checkpoints/multitask_6/policy/flow-20/flow_multi_ep0100_20260320_114720.pt` | flow_multi ckpt |
| `DYN_CKPT` | `""` | trained dynamics predictor (optional) |
| `OMEGA` | `0.5` | dichotomous weight; 0 = KNN-only |
| `K` | `1` | k-NN order |
| `BETA`, `KAPPA` | `auto` | F3 params; `auto` = 1/MAD, median |
| `SIGMA_SQ` | `0.5` | weighted-KNN bandwidth |
| `DELTA` | `10.0` | tau percentile FA budget |
| `SUCC_NUM`, `FAIL_NUM` | `200`, `100` | per-task eval caps (map to --max-*-per-task) |
| `SHARE_BANKS` | `1` | D8 shared banks; 0 = per-task |
| `NO_NORMALIZE_FEATURE` | `0` | only affects projected phi |

### `train_d3_dynamics.sh`

| Env var | Default | Meaning |
|---|---|---|
| `TASKS` | 6 canonical envs | expands to `data/<task>/{expert,success_rollout}` paths |
| `EXPERT_NUM`, `SUCC_NUM` | `0`, `0` | per-kind trajectory cap (0 = all); max wins for `--max-trajectories-per-kind` |
| `EPOCHS`, `BATCH_SIZE` | `30`, `256` | |
| `D_MODEL`, `NUM_LAYERS`, `NUM_HEADS` | `512`, `6`, `8` | predictor arch |
| `HORIZON` | `1` | |
| `EXPERT_RATIO` | `0.5` | weighted sampler expert/rollout mix |
| `PROPRIO_LOSS_WEIGHT` | `0.1` | |
| `VAL_RATIO` | `0.05` | |
| `SAVE_FREQ` | `10` | periodic ckpts every N epochs |

---

## 5. Known quirks / gotchas for future agents

1. **Task name for cache key**: `FlowMultiEncoderWrapper.cache_key(task_name, file_path, demo_key)` uses `task_name` verbatim. The cache was written with `PickPlaceBread`, `PandaPickPlaceCan`, `PandaLift`, etc. — matches `BenchmarkTrajectory.task_name`. `_TASK_TO_CKPT` in `encoder.py` maps canonical -> checkpoint-side (`PandaPickPlaceCan` -> `PickPlaceCan`) only for `task_metadata_map` / `task_prompt_map` lookups.
2. **Dataset infers task from path**: `_infer_task_name(fp)` takes the grandparent dir, i.e. `data/<task>/<kind>/<file>.hdf5`. Moving HDF5 files out of that layout breaks cache key generation.
3. **The trainer loads latents via `FlowMultiEncoderWrapper.load_or_encode_demo`**: on cache MISS it will fire up flow_multi and encode from scratch. Cache is shared with the benchmark pipeline.
4. **Proprio handling**: raw state from HDF5 is right-padded / truncated to `self._proprio_dim = max state_dim seen` (LPB-style). With `proprio_indices`, a fixed slice is used. Default: all state dims (right-padded across tasks).
5. **PandaLift fail_rollout**: does not exist. Discriminator degrades gracefully (contributes only to `D+`). Fit skips if no success calib for a task.
6. **PickPlaceCan source HDF5**: some cache entries point to paths that may no longer exist on disk. The `load_or_encode_demo` will re-encode on MISS but a missing source HDF5 raises.
7. **ω=0 sanity**: with `DYN_CKPT` empty, ω=0 is effectively LPB on raw-z features (not identical to LPB's own ResNet+dynamics). Don't expect byte-level parity with `LPBBenchmarkDiscriminator`.
8. **cuDNN determinism**: set `torch.backends.cudnn.deterministic=True`, `benchmark=False` when bit-exact reproduction against the cache matters. Default flags + `encoder_batch_size=256` give ≤ 5e-6 drift on 5/6 tasks.
9. **Out of scope** (do NOT add without explicit user approval):
   - K-step rollout through `f_phi` (phase 2).
   - Iterative F3-EM (D11).
   - F1 (GT-mask) / F2 (expert-only-dynamics) filters (D3).
   - Retraining flow_multi.
   - Regenerating the cache.
   - Modifying `robosuite/discriminator/lpb/`.

---

## 6. Hand-off pointers

- Design spec: `.claude/discriminator/sfv_discriminator_design.md`
- Implementation plan: `.claude/discriminator/implementation_plan.md`
- Prior context: `.claude/discriminator/conversation_context.md`
- Approved plan for this session: `~/.claude/plans/you-are-implementing-d-disc-polished-kitten.md`
- Entry points: `robosuite/discriminator/d3disc/scripts/*.sh`, `data/utils/benchmark/examples/run_d3.py`
- AGENTS.md conventions apply (English comments, WANDB offline, `daggar` conda env, minimal modifications).
