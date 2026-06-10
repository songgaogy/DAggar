# IQL Q-Chunking Critic — `dipole-rl/v4-expo` Working Notes

Context summary for the offline IQL Q-chunking critic work on branch
`dipole-rl/v4-expo`. Task: PickPlaceCereal. Python env: conda `dagger`
(`/home/dodo/miniconda3/envs/dagger/bin/python`).

---

## 1. Goal

Bring EXPO-FT's critic-input tricks into the existing offline IQL Q-chunking
learner (`robosuite/pipeline/algorithms/q_learning/`), then diagnose why the
learned Q/V is poorly shaped on PickPlaceCereal demos and decide on fixes.

---

## 2. Code changes already landed (branch `dipole-rl/v4-expo`)

### 2.1 Network: bare concat + scale matching (`networks.py`)
- **Removed** the `compressed_dim` context bottleneck (`_build_context_compressor`,
  `compress_net`). Q now concatenates the **full** `D_ctx` context directly
  ("bare concat"). `_input_dim = context_dim + action_dim * action_horizon`.
- **Scale matching (EXPO-style)** inside `QChunkNetwork.forward`:
  - context branch is `tanh`-bounded to `[-1, 1]`.
  - action chunk is **quantile-normalized** (`q01`/`q99` → `[-1, 1]`) via
    registered buffers `act_q01` / `act_q99` (default identity = `-1`/`+1`).
- **Activation** in `_build_mlp`: `GELU` → `ReLU` (matches EXPO critic
  `Linear→LayerNorm→ReLU`). Final Linear stays zero-init.

### 2.2 Learner (`iql.py`)
- Dropped all `compressed_dim` plumbing (ctor arg, `state_dict`,
  `load_state_dict` check).
- Added `set_action_norm_stats(q01, q99)` (broadcasts to every ensemble member)
  and `action_norm_stats` property.
- Quantile buffers live in `q_ensemble.state_dict()`, so they are saved/loaded
  automatically — the vis path reuses the exact same normalization.

### 2.3 Config / common
- `common.py`: removed `compressed_dim` field + validation from `IQLConfig`.
- `train_dipole_rl.yaml`: removed `compressed_dim`; **`reward_mode: "-1/0" → "0/1"`**.

### 2.4 Action quantile stats (`data_util.py`, `warmup.py`)
- `compute_action_quantile_stats(actions, q_low=0.01, q_high=0.99)` (degenerate
  dims widened by ±0.5).
- `warmup.py` gathers actions from the offline buffer, computes q01/q99, calls
  `iql.set_action_norm_stats(...)` **before** training, and saves stats both:
  - embedded in `iql_state.pt` payload (`action_norm_stats`, `schema_version=2`), and
  - a sibling `action_norm_stats.pt` in the same directory.

### 2.5 BON diagnostic now ranks by Q-min (`utils/vis_qv.py`)
- Candidate ranking / best / delta / summary / CSV / plots switched from
  `q_mean` → **`q_min` over the full ensemble** (pessimistic, suppresses OOD
  overestimation). Keys renamed: `candidate_rank_by_q_min`, `delta_q_min_vs_demo`,
  `*_q_min_*`, `ranking_score="q_min"`.

### 2.6 Debug tool
- `robosuite/pipeline/scripts/utils/debug_success_done.py`: replays stored
  state+action through the env (skips image IO) to verify the loader's
  done/reward logic.

All edits pass lint; `tests/test_iql.py` (13 tests) passes; state_dict
round-trip + normalization smoke-tested.

---

## 3. EXPO trick adoption — assessment

**Adopted (critic input / anti-collapse):** bare concat (no action encoder, no
bottleneck), tanh-bounded context + quantile-normalized action (scale match),
`Linear→LayerNorm→ReLU` MLP, Q-ensemble + subset-min (REDQ-flavored), `γ^H`
chunk bootstrap, Polyak target.

**NOT adopted (algorithm paradigm — intentionally):** EXPO's value-greedy target
(REDQ `min` over sampled base-policy + residual-edited actions, i.e. max-Q via
best-of-N **inside** the backup), the residual SAC actor, REDQ scale
(`num_qs=10, num_min_qs=2, utd=20`), `replan_steps` decoupled from
`action_horizon`. The user's learner stays IQL V-bootstrap (offline-safe);
EXPO's max-Q-in-backup would break offline safety, so it was deliberately left
at the policy-extraction layer only.

---

## 4. Debug result: success demos DO get terminal done + reward 1

`reward_mode=0/1` replay over **all 200** PickPlaceCereal success demos:
`done+reward1 = 200/200, no_done = 0`. Success re-fires on replay at
`first_success_idx ≈ 165–360`, then the loader sets `done=True`, `reward=1.0`,
and breaks → loaded transitions = `success_idx+1` (not the full 400). **The
done/reward path is correct; there is no missing-done bug.**

---

## 5. Results analysis (PickPlaceCereal)

### 5.1 `-1/0` reward (`...tau07-vis`, earlier run)
- `γ=0.99` ⇒ floor `-1/(1-γ) = -100`. Saturated steady state in the reward-free
  region: `V* = -7.726/(1-0.99^8) ≈ -99.95`. So the first ~250 steps sit near
  the floor **by construction** (a perfect critic gives ≈ -92 at the start of a
  254-step success). Front "too low" ≈ correct given γ, but useless as signal.
- Target imbalance: ~95% of targets ≈ floor → critic biased to floor,
  under-fits the rare terminal climb. seed42 last window: target -6.79 but
  Q=-81 (TD residual ~74) → "success but Q/V stuck near floor".
- BON: demo beaten by random (`demo_rank1=0.008`, `best_margin_max=+49`) →
  classic OOD overestimation (IQL doesn't constrain Q on OOD actions).

### 5.2 `0/1` reward (`...tau07-01reward-vis`, current run)
demo_000015, success@~201, last window step 194 (`done=1`):

| quantity | value | note |
|---|---|---|
| V(start) | 0.011 | true `γ^194≈0.142`; critic an order too low |
| last window Q_mean / Q_max | 0.93 / 0.99 | target 0.932 (=`γ^7`) → **Q is on target** |
| last window V | 0.81 | below Q (expectile_tau=0.7 drags V down) |
| steps 186→192 | 0.79→0.18, then 0.93@194 | strongly non-monotonic |
| TD residual mean | 0.015 | Bellman essentially satisfied |
| advantage range | [-0.11, +0.12] | ≈0 in the front |

### 5.3 Root causes (why 0/1 didn't fix 1/2/3)
The TD residual ≈ 0.015 means the critic converged to a **self-consistent but
badly-shaped fixpoint**. Reward sign is not the bottleneck. Three deeper causes:
1. **γ=0.99 exponential compression** → `V=γ^{steps_to_go}` is convex: flat
   near-0 front, steep only in the last ~50 steps (problem 1).
2. **Value aliasing + frozen discriminator encoder** → early success vs fail
   frames look alike ⇒ V regresses low; adjacent frames jitter (0.79→0.18) ⇒
   non-monotonic (problems 1-level-low and 3). Bootstrapping on a jittery `V'`
   yields a bumpy but low-residual fixpoint.
3. **Vis windowing** stops at `len-H`; the final H-1 frames (where V should
   climb 0.93→1.0) are never plotted, and V(terminal state) is never queried.
   So "end not ≈1" is partly a measurement artifact — Q already hits 0.93/0.99;
   V lags only because of expectile_tau and the unplotted tail (problem 2).

### 5.4 BON (problem 4) — already much better with `q_min` + `0/1`
`best_margin_q_min_over_demo_max`: **49 → 0.26**; random mean delta **-0.16**
(clearly worse); `demo_top3`: 0.097 → 0.34. Remaining "random sometimes wins by
~0.04" is just the flat-front (advantage ≈ 0) regime → fixing problem 1 helps.

---

## 6. Proposed fixes (ranked; focus on problems 1 & 2)

- **A. Increase γ → 0.995–0.997** (config + re-warmup). With success@194:
  `γ^194`: 0.99→0.14, 0.995→0.38, 0.997→0.56; end `γ^7`: 0.99→0.93, 0.997→0.98.
  Lifts/linearizes the front and pushes the end toward 1. Helps 1/2/3/4.
- **B. `expectile_tau` 0.7 → 0.9** so V tracks the upper Q (fixes "V < Q" at the
  end; less dragged by aliased low neighbors). Directly helps problem 2.
- **C. Dense / potential-based progress reward** from the LPB discriminator
  (`disc_reward_coef>0`, ideally `F=γΦ(s')−Φ(s)`). De-flattens the front,
  smooths shape; decouples from γ. (Design caveats in §7.)
- **D. Vis fix**: evaluate the terminal / shrinking final windows (or
  `V(terminal)`) so the last frames (V→1) are actually plotted. Problem 2 is
  partly a measurement artifact.
- **E. Representation**: frozen discriminator features don't support a smooth
  monotone value. Options (light→heavy): trainable adapter on frozen context;
  concat proprio/time features; unfreeze/fine-tune encoder. (Design in §7.)
- **F. BON / OOD**: optional CQL-style OOD penalty or restrict candidates to
  near-data (EXPO residual) instead of uniform random.

**Recommended first experiment:** A+B+D (cheap), then C if the front is still
flat; use E only if the curve stays jittery after A+B (that isolates it as a
frozen-feature problem).

---

## 7. Design notes for the "fundamental" fixes

### C. Dense progress reward (LPB)
- Prefer **potential-based** `r' = r_env + (γΦ(s') − Φ(s))` (policy-invariant)
  over `r_env + λΦ(s)`.
- **Validate Φ monotonicity first**: LPB outputs a failure/expert logit, not a
  guaranteed-monotone progress signal; plot Φ along success/fail trajectories
  before wiring it in (else it amplifies the jitter of problem 3).
- Respect the chunk `γ^H` bootstrap: the potential telescopes to
  `γ^H Φ_H − Φ_0` over a chunk — implement carefully vs `aggregate_chunk_reward`.
- Offline-safe (Φ is a frozen scorer).

### E. Representation
1. **Trainable adapter** (lightest, recommended): small MLP/LayerNorm on top of
   the frozen context, trained with the value loss. Reshapes feature geometry
   without breaking the discriminator representation.
2. **Concat explicit features**: proprio / normalized timestep (timestep helps
   "steps-to-go" values most; ensure vis/online parity).
3. **Unfreeze/fine-tune encoder**: most expressive, highest risk (LPB reward
   still depends on it; offline fine-tune can overfit). Avoid unless needed.

---

## 8. Current status / open decision

- A/B/C/D/E **not yet implemented** — user is deciding (chose "hold" to think
  about C/E design).
- Offered (not yet done): a **read-only diagnostic** plotting `Φ(s)` (LPB score)
  and a "progress separability" curve of the frozen context along success
  trajectories, to decide whether C's Φ is monotone enough and whether E is
  necessary before changing reward/representation.

---

## 9. Key file map

| File | Role |
|---|---|
| `algorithms/q_learning/networks.py` | `QChunkNetwork` (bare concat, tanh ctx, quantile action norm, ReLU MLP), `VNetwork` |
| `algorithms/q_learning/iql.py` | `IQLLearner` (V-bootstrap target, expectile V, ensemble subset-min, norm-stat API) |
| `algorithms/q_learning/common.py` | `IQLConfig` (reward_mode, discount, expectile_tau, ensemble sizes) |
| `algorithms/q_learning/data_util.py` | `aggregate_chunk_reward`, `chunk_done_mask`, `compute_action_quantile_stats` |
| `algorithms/q_learning/warmup.py` | offline warmup: load demos, compute/set/save action norm stats, train, dump `iql_state.pt` |
| `algorithms/q_learning/utils/vis_qv.py` | Q/V + BON visualization (BON ranks by q_min) |
| `scripts/utils/debug_success_done.py` | replay-verify success done/reward |
| `scripts/utils/init_iql_qv.sh`, `vis_iql_qv.sh` | warmup / vis entrypoints |
| `config/train_dipole_rl.yaml` | IQL config (`reward_mode: "0/1"`, `discount: 0.99`, `expectile_tau: 0.7`, …) |
