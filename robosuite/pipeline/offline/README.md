# Offline DIPOLE

Reproduce a **single online `policy → interaction → policy update` step** from a fixed, pre-collected dataset — so the DIPOLE update can be validated/tuned without a human continuously interacting with the environment.

Two stages: **(1) collect** a dataset by running the fixed policy + discriminator with human intervention (no updates), then **(2) offline-train** DIPOLE (finetune IQL, then a routed weighted-BC policy update) on that dataset.

---

## Stage 1 — Collect data (`[done]`)

`scripts/collect_data.sh` → `utils/collect_data.py`. Mimics the online rollout loop (policy rollout + nnPU discriminator HUD + SpaceMouse/keyboard human intervention) but with a **fixed policy and no IQL / no policy update**.

Saves one file per task to `data/<task>/offline_data/offline_episodes.pt` (+ `.meta.json`). Payload is `{**metadata, "episodes": [...]}`; each episode holds parallel per-step arrays:

| field | meaning |
|---|---|
| `obs` / `next_obs` | `{"state": (T,D), <camera>: (T,H,W,3) uint8}` |
| `executed_action` | action actually stepped (policy, or human during intervention) |
| `policy_action` | policy's inferred action **every** step (incl. during intervention) |
| `human_action` | SpaceMouse override (zeros when not intervening) |
| `is_intervention` | per-step human-intervention label |
| `reward` / `success` / `done` | as collected (sparse success) |
| `terminal_reason` | `success` / `env_done` / `max_steps` / `manual_reset` / `interrupted` |

```bash
bash robosuite/pipeline/offline/scripts/collect_data.sh   # edit TASK / POLICY_CKPT / NNPU_CKPT inside
```

## Stage 0 — Pretrained IQL critics (prerequisite)

Offline training **continues** from a warmed-up IQL. Produce it once with `../scripts/utils/init_iql_qv.sh` (writes `iql_state.pt` and, with `warmup.num_trajectories.save_data=true`, the warmup transitions `data/<task>/offline_data-iql/iql_offline_transitions.pt` that get mixed back in).

---

## Stage 2 — Offline DIPOLE (implemented)

`scripts/train_offline_dipole.sh` → `src/train_offline_dipole.py` (Hydra config `config/train_offline_dipole.yaml`, inherits `train_dipole_rl`).

**Sequential two phases:**

- **Phase A — IQL finetune (unfrozen).** Load the pretrained IQL (V-only N-head soft-LCB ensemble, expectile-TD), continue training it (`IQLLearner.update`) on the collected policy sections **mixed with** the warmup transitions, save `<run_dir>/checkpoints/iql_state_finetuned.pt` (schema v5). A one-shot frozen-encoder feature cache (`preencode_step_cache`) keeps the loop encoder-free. The value-semantics config (`expectile_tau`, `ensemble_lcb_beta`, `ensemble_bootstrap_prob`, `discount`, reward coefs) is aligned from the loaded checkpoint so finetuning continues in the warmup regime; structural fields (`v_ensemble_size`, projector dims) are asserted by `load_state_dict`.
- **Phase B — weighted-BC policy update (IQL frozen).** Precompute the per-window advantage against the frozen finetuned critics on the **soft-LCB value** `V_lcb`, then update the two flow policies with a pluggable routed branch-weight policy. The 1-step (macro-step) TD residual is `A = r + γ^H·(1−done)·V_lcb_target(s') − V_lcb(s)`. `offline.advantage.estimator` selects the read-out: `gae` (default) accumulates GAE(`gae_lambda`, default 0.6) backward over each policy-section episode (`A_t = δ_t + γ^H·λ·(1−done_t)·A_{t+H}`, matching `q_learning/utils/vis_qv.py`); `td1` uses the bare residual. Sections are standalone short episodes with `done=True` on the last frame, so GAE never bootstraps across a section boundary.

### Data usage — split each episode on `is_intervention`

`utils/episode_dataset.build_offline_transitions` produces three routed streams (each frame tagged `info["route"]`):

| stream | frames | BC target | route | used by |
|---|---|---|---|---|
| **policy** | policy sections (`is_intervention=False`) | `executed_action` | `advantage` | policy update (advantage-weighted) **and** IQL |
| **human** | human sections (`is_intervention=True`) | `executed_action` (=human) | `pos_only` (`w_pos=1`) | pos policy only |
| **neg** | same human frames (default on) | `policy_action` | `neg_only` (`w_neg=1`) | neg policy only |

Rules:
- **Keep** a policy section only if it ends in success / human-intervention / `manual_reset`; drop otherwise.
- **Rewards** (policy sections, `offline.reward_success` / `reward_fail`): per-frame constant by outcome — success section → `0` (with per-frame `info["success"]` → IQL absorbing terminal); other → `−1` (`done` at section end → IQL truncation terminal). Intervention frames carry no reward (not in IQL).
- **Chunk boundary** (each section is its own `episode_index`, so windows never cross section boundaries): human/neg sections shorter than `H` are **padded** to `H` (BC-only); policy sections shorter than `H` are **dropped** (never fabricate `s'` for IQL).

### Branch weights (modular)

`utils/branch_weights.RoutedSigmoidBranchWeightPolicy` (default) reads `batch.metadata["route"]`: `advantage → w_pos=σ(β·(G+k)), w_neg=1−w_pos`; `pos_only → (1,0)`; `neg_only → (0,1)`. Swap the format via `offline.branch_weight.type` (e.g. `disc_scaled`, an example extension) — the policy is attached with `DipoleFlowPolicy.set_branch_weight_policy`, which short-circuits the built-in weighting. `k` offsets `G` before scaling (`w_pos=0.5` at `G=-k`); `beta` is the post-offset slope (`DipoleFlowPolicy._g_weights_from_raw`).

### Run

```bash
bash robosuite/pipeline/offline/scripts/train_offline_dipole.sh
# quick smoke:
NUM_TRAIN_STEPS=20 IQL_FINETUNE_STEPS=20 BATCH_SIZE=8 \
  bash robosuite/pipeline/offline/scripts/train_offline_dipole.sh
```

Outputs land in `outputs/dipole-rl-offline/<task>_<timestamp>_<postfix>/`: `checkpoints/{latest,step_*}.pt`, `checkpoints/iql_state_finetuned.pt`, `tensorboard/` (Phase A under `iql_finetune/*`, Phase B under `train/*`), `run_info.json`, resolved config.

### Evaluate

`scripts/eval_offline_dipole.sh` → `src/eval_offline_dipole.py`: headless success-rate + video with two-branch CFG guidance (`OMEGA`). Point `POLICY_CKPT` at a finetuned checkpoint, e.g. `outputs/dipole-rl-offline/<task>_<ts>_<postfix>/checkpoints/latest.pt`.

```bash
bash robosuite/pipeline/offline/scripts/eval_offline_dipole.sh   # set POLICY_CKPT / OMEGAS
```

---

## Key config knobs (`config/train_offline_dipole.yaml`)

| key | default | meaning |
|---|---|---|
| `offline.episodes_path` | `null` → `data/<task>/offline_data/offline_episodes.pt` | collected dataset |
| `offline.iql_warmup_transitions_dir` | `offline_data-iql` | warmup transitions dir mixed into IQL |
| `offline.reward_success` / `reward_fail` | `0.0` / `-1.0` | per-frame section-outcome reward |
| `offline.include_policy_action_neg` | `true` | route `policy_action` → neg branch |
| `offline.iql_finetune.{num_steps,value_only_steps,batch_size,preencode_cache}` | `20000,0,256,true` | Phase A |
| `offline.num_train_steps` | `15000` | Phase B policy steps |
| `offline.advantage.{estimator,gae_lambda}` | `gae,0.6` | Phase-B read-out: `gae` (GAE over section) or `td1` (1-step residual) |
| `offline.branch_weight.{type,beta,k}` | `routed_sigmoid,2.0,0.0` | `w_pos=σ(β·(G+k))`. `G` is **not** normalized: `beta` is post-offset slope (`~2.0` for `gae` λ=0.6 ≈ 2× 1-step scale, `~4.0` for `td1`); `k` shifts the threshold in G-space. Tune against the branch-weight histogram. |
| `algorithm.advantage_g_provider.{alpha,beta}` | `1.0,0.0` | `G = alpha·A − beta·failure` (β=0 → pure advantage) |

## File map

```
offline/
├── scripts/           collect_data.sh · train_offline_dipole.sh · eval_offline_dipole.sh
├── src/               train_offline_dipole.py (Phase A→B) · eval_offline_dipole.py · diagnostic_plots.py
└── utils/
    ├── collect_data.py        Stage-1 collector (fixed policy, no updates)
    ├── episode_dataset.py     split on is_intervention → 3 routed Transition streams
    ├── branch_weights.py      pluggable RoutedSigmoid / DiscriminatorScaled branch weights
    ├── iql_finetune.py        Phase-A mixed buffer + unfrozen IQL update + save (schema v5)
    ├── advantage.py           precompute_offline_advantage (td1 | gae read-out) + OfflineAdvantageGProvider
    └── setup.py               build_agent_env / finalize_normalizers / make_hdf5_loader
```
`route` is threaded through `algorithms/dipole/{common.py, models/flow.py, replay_buffer.py}`.

> Note: the warmup transitions file can be large (~16 GB for PickPlaceCereal); full mix + preencode is memory- and time-heavy. A subsample cap is a sensible future addition for fast iteration.
