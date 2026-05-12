# Failure-Aware Discriminator: Background, Status, and Ideas

Working notes for the next iteration of the robosuite/AgileX failure discriminator. The current LPB v2 baseline lives in `robosuite/discriminator/lpb_v2/`; baselines we compare against are in sibling folders (`float/`, `logpZO/`, `lpb/`).

---

## 1. Problem statement

We want a per-frame **failure detector** that works on (a) robosuite simulation rollouts and (b) self-collected AgileX real-world rollouts. The detector must return a score and a binary failure prediction at each timestep of a trajectory.

Operating constraints that drive the design:

- **Failure data is scarce.** Most policy rollouts in our corpus succeed.
- **Frame-level failure ground truth is expensive.** A failure trajectory only carries a `first_gt_failure_frame` / `failure_segments` annotation, and even that is noisy: the frames *before* the GT mark are visually indistinguishable from success demonstrations because the robot has not yet diverged.
- **Real-world is the hard target.** All existing methods we benchmarked are acceptable in robosuite but weak in real-world AgileX rollouts.

## 2. Existing methods (under `robosuite/discriminator/`)

| Folder      | Method                  | Reference                              | One-line summary                                                                 |
|-------------|-------------------------|----------------------------------------|----------------------------------------------------------------------------------|
| `float/`    | FLOAT                   | https://arxiv.org/pdf/2510.02298       | DINO-style encoder + scoring head.                                               |
| `logpZO/`   | logpZO                  | https://arxiv.org/pdf/2503.08558       | Normalizing-flow density on a frozen visual feature; score = `-log p(z)`.        |
| `lpb/`      | LPB (original)          | https://arxiv.org/pdf/2508.05941       | Latent-prediction visual dynamics + KNN over `[visual; proprio; action]` features. |
| `lpb_v2/`   | **LPB v2 (ours)**       | this repo                              | WAM-style latent-dynamics pretraining + KNN (or transformer-hidden-state KNN) on success demos. |

All four are **success-only / OOD-style**: they fit a representation or density over success data and treat "far from success" as failure. None of them currently consume failure annotations during training.

## 3. LPB v2 (current SOTA in sim) — what is actually implemented

Implementation lives in `robosuite/discriminator/lpb_v2/`. The pipeline has two stages:

### 3.1 Pretraining: latent visual dynamics (WAM-style)

`train.py` + `models/visual_dynamics.py`:

- ResNet per camera view → tokens; MLP proprio encoder; MLP action encoder.
- Tokens are fed into a temporal ViT predictor.
- Loss = MSE between predicted and target latent tokens (visual + proprio, optionally action).
- Output of a run directory:
  - `hydra.yaml` — resolved config (needed to reload).
  - `normalizer.pth` — image/state/action normalization.
  - `checkpoints/model_<epoch>.pth` — encoder + predictor + proprio/action heads.

### 3.2 Discrimination: KNN over success demos

`knn.py`:

- `LPBV2Encoder` reloads the checkpoint and applies the same normalization / crops as training. It exposes two `feature_source` modes:
  - `encoder`: `[visual_emb ; proprio_emb ; action_emb]` with per-block weights.
  - `transformer`: flattened ViT hidden state at a chosen layer (uniform L2).
- `LPBV2KNN.fit(expert_features, calibration_features)` builds a feature bank from success demos and calibrates a percentile threshold `tau` on a *disjoint* calibration split of success demos.
- Inference: `score_t = min_j ||feature_t - bank_j||_2`; predict failure when `score_t >= tau`.

### 3.3 What this *doesn't* do

These are the levers we are currently leaving on the table:

- **Failure data is never used** — neither for representation learning nor for threshold/score calibration.
- **The predictor is unused at inference.** Training fits a one-step latent predictor, but KNN scoring throws it away and only consumes the encoder output. The latent dynamics prediction error is a free signal we currently discard.
- **Per-frame independence.** KNN scores each frame in isolation; no temporal smoothing, no trend modeling. A momentarily noisy frame can flip the binary prediction even when the trajectory as a whole is fine.
- **Uniform per-dim weighting.** The metric is L2 with fixed block weights; we never *learn* which latent dimensions matter for separating failure from success.

## 4. Real-world latent visualization (2026-05-11)

Run: `checkpoints/lpb_v2/real_world_eval/run_20260511_170606_vis/`

- Encoded 200 trajectories (100 success / 100 failure) across 4 AgileX tasks (`candy_in_plate`, `duck_in_bowl`, `Micky_in_box`, `sausage_in_pot`).
- Each frame labeled into one of three phases:
  - `success` — every frame of a success rollout.
  - `failure_before_gt` — frames before `first_gt_failure_frame` of a failure rollout.
  - `failure_after_gt` — frames from `first_gt_failure_frame` onward.
- Outputs: `latent_pca.png`, `latent_tsne.png`, `latent_points.npz`, `latent_points_meta.json`.

### What the plots show

- **PCA** (linear projection): `failure_after_gt` (red) forms a dense cluster on the left, but `success` (light blue) is spread across the **whole** 2D projection — including the same left region red occupies. The two classes **overlap; they are NOT linearly separable.** `failure_before_gt` is also scattered, as expected.
- **t-SNE** (local-neighborhood-preserving): `success` and `failure_after_gt` form mostly distinct small islands with little overlap. `failure_before_gt` is the only phase that still overlaps everywhere.

### Interpretation

The `failure_before_gt` overlap with `success` is **expected and correct**: those frames are visually still success-like — the robot has not yet diverged.

The interesting pair is `success` vs `failure_after_gt`. The combination "PCA overlaps but t-SNE separates" is a strong signal that:

- **The information is in the latent**, but it is not aligned with any high-variance linear direction. PCA finds the directions of maximum variance — those are dominated by *task / view* variance, not failure variance.
- **The separability is local and nonlinear.** t-SNE preserves local neighborhoods, so success frames have systematically different *neighbors* than failure frames, even when their global PCA positions overlap.

**Implications for scoring:**

- A **linear probe** on the raw latent will **not** work. PCA is already the optimal linear projection, and it overlaps. Any logistic regression / linear SVM on these features will hit the same wall.
- KNN actually fits this setting better than I first credited it: KNN is a local method, and "local-neighborhood-separable" is exactly the regime where KNN can win — *as long as the distance metric isn't dominated by irrelevant variance*. The likely reason real-world KNN underperforms sim KNN is that real-world latents carry more irrelevant variance (lighting, background, view jitter) that drowns out the failure signal in plain L2.
- The most promising directions are therefore **(a) learn a metric / weighting on the frozen latent** so KNN's L2 sees the right dimensions, or **(b) a nonlinear supervised head** (MLP). Both can directly exploit what t-SNE already shows is there.

## 5. Why naive failure supervision is dangerous

Two non-obvious failure modes if we just label and train:

1. **Wrong labels in the "before GT" region.** Treating every frame of a failure trajectory as positive will teach the model that perfectly valid robot behavior is a failure (because the same approach motion appears in both success and failure rollouts early on). This collapses precision.
2. **GT mark is noisy itself.** `first_gt_failure_frame` is when failure *becomes visible to a human annotator*, not when the underlying cause started. Treating it as a sharp boundary inflates label noise near the transition.

The safe primitives we *can* trust are:

- Trajectory-level labels (`is_failure ∈ {0, 1}`) — cheap, certain.
- "Definitely failure" tails (frames well after `first_gt_failure_frame`, e.g. the last K frames) — also nearly certain.
- "Definitely not failure" (all frames of a success rollout) — certain.

Anything close to the GT boundary or before it should be **treated as unlabeled or down-weighted**, not as positive supervision.

## 6. Ideas to evaluate (prioritized)

Two ideas that previously looked attractive are now ruled out and listed in §6.0 so we don't re-propose them. Real candidates start at §6.1.

### 6.0 Rejected (do not revisit without new evidence)

- **Use the WAM predictor's latent prediction error as a score channel.** Empirically does not work on this corpus. The dynamics predictor is trained on too little data to be reliable, so `||ẑ_{t+1} - z_{t+1}||` carries mostly noise. Only the encoder side of the WAM checkpoint is treated as useful.
- **Temporal smoothing / EMA on the per-frame score.** A small engineering trick, not a methodological improvement. Not a path to closing the real-world gap.

### 6.1 Idea — Two-bank KNN: success **and** failure as reference sets *(do first)*

**Motivation.** Right now scoring is asymmetric: `score = min_j ||x − success_bank_j||`. It only knows "how far from success." But t-SNE shows `failure_after_gt` forms its own locally-distinct cluster — we have actual failure points in feature space and we are throwing them away at inference. Adding a second bank turns the detector from one-class OOD into a relative comparison between two empirical distributions, **without training anything**.

**Plan.**

1. Build `success_bank` exactly as today (success demos, expert split).
2. Build `failure_bank` from the **last K frames** of each failure trajectory (K conservative, e.g. 30–60). Skip `failure_before_gt` entirely.
3. Per-frame scoring options to compare:
   - **Difference:** `score = min_j ||x − success_bank_j|| − min_k ||x − failure_bank_k||`
   - **Ratio:** `score = d_succ / (d_succ + d_fail)`
   - **2D probe:** treat `(d_succ, d_fail)` as a 2D feature and fit a 1-line decision (logistic or threshold on `d_succ − α · d_fail`).
4. Calibrate the threshold on a **disjoint** success+failure split, not just success. (Current percentile-on-success calibration cannot use failure points; the new setup unlocks that too — see §6.4.)

**Why it survives "data too small."** Zero parameters learned. Failure data is used only as a reference set, exactly like success demos already are.

**Composability.** Orthogonal to §6.2 (learned metric) and §6.3 (encoder swap) — can stack.

**Cost.** Drop-in change to `LPBV2KNN`. No training. Minor change to benchmark wiring.

### 6.2 Idea — Learned metric / projection on the frozen latent, then two-bank KNN

**Motivation.** t-SNE separates success from `failure_after_gt` but PCA does not. That means the discriminative information lives in a *non-axis-aligned, non-top-variance* subspace of the WAM latent, which plain L2 cannot see. A small learned projection that aligns success-vs-failure-after-gt structure with its axes makes that subspace explicit *without changing the encoder* — and is the most data-efficient supervised step we can take, which matters because failure data is small.

**Plan.**

1. Freeze the WAM encoder.
2. Anchor labels: success frames as one class; the **last K frames** of each failure trajectory as the other (same safe set as §6.1). `failure_before_gt` excluded.
3. Learn a linear projection `W ∈ R^{d×D}` (d ≈ 64–128). Loss: supervised contrastive / triplet / NCA — choose for ease, not novelty.
4. Apply `W` to all features and re-run the two-bank KNN from §6.1 in the projected space.

**Why metric learning rather than an MLP head:** A `d × D` linear projection has ~10⁴–10⁵ parameters; a 2-layer MLP head is an order of magnitude more, and overfitting risk on a few thousand failure frames is high. Metric learning also preserves the non-parametric KNN at inference, so it composes cleanly with §6.1 and §6.3.

**Cost.** Single-GPU minutes. Frame-level annotation as it already exists in `failure_segments`. Drop-in for `LPBV2KNN`.

### 6.3 Idea — Swap the encoder for a frozen foundation model

**Motivation.** The user's own observation: the WAM-style pretraining gives a usable latent but the dynamics learning is **bottlenecked by dataset size**. If small data is the root cause, the cleanest fix is to stop relying on our own data to learn the visual representation at all. Use a frozen pretrained vision encoder — DINOv2, V-JEPA / V-JEPA-2, SigLIP, Theia, etc. — and keep the proprio/action MLPs and downstream KNN pipeline.

**Plan.**

1. Replace the per-view ResNet (`models/resnet_encoder.py`) with a frozen foundation backbone; pool tokens into a fixed-size visual embedding per view.
2. Keep proprio / action encoders as today; concat as the KNN feature, exactly like current `feature_source="encoder"`.
3. Re-run the same KNN (eventually two-bank, §6.1) and threshold calibration.

**Risks worth checking on the actual data.**

- Frozen foundation models were not trained on our robot views; the embedding may carry less *task-discriminative* signal than the WAM-trained ResNet, even if it generalizes better.
- The current proprio/action concat sits next to the visual block; weighting may need re-tuning when the visual dim changes.

**Why now.** `float/` already runs on DINO-style features, so part of the plumbing exists in this repo. This is one of the few moves that addresses "data too small" at its root rather than working around it.

### 6.4 Idea — Calibrate the threshold using failure data, not percentile-on-success

**Motivation.** Currently `tau` is set as a percentile on the success calibration split; this never looks at a single failure frame. We have ~100 failure rollouts in the real-world set — small, but enough to set a per-task threshold by maximizing F1 / Youden's J on `(d_succ_calib_succ, d_succ_calib_fail_after_gt)` pairs, or equivalently on the two-bank score from §6.1.

**Why it's not "just a trick."** Unlike temporal smoothing, this changes *what information defines the decision boundary*. Today's threshold is a quantile of a one-class density; the new one is a discriminative cut on actual two-class data. It is the simplest way to inject failure supervision and it is required to evaluate §6.1 fairly.

**Cost.** Pure scikit-learn level code. Pairs naturally with §6.1.

### 6.5 Idea — MIL on trajectory-level labels (fallback for label noise)

**Motivation.** If §6.2's "last K frames" anchor set turns out to be a bad approximation of "actually failing" on some tasks, drop frame-level supervision entirely and use only `is_failure`. MIL pools per-frame predictions into a trajectory prediction:

- Success trajectory: top-k frame scores must stay low.
- Failure trajectory: top-k frame scores must reach high.

**Plan.** Noisy-OR or top-k pooling loss over per-frame outputs of the §6.2 projection (or a small head). No `first_gt_failure_frame` required.

**Cost.** More moving parts than §6.2. Reserve for the case where §6.2 underperforms on a task and inspection points at label noise as the cause.

### 6.6 Idea — Contrastive fine-tuning of the encoder (last resort)

Touches the encoder, conflicts with the "data is too small to train representation" premise, breaks the frozen-latent contract that §6.1–§6.2 rely on. Listed only for completeness; defer unless §6.1–§6.5 collectively plateau.

## 7. Suggested experimental order

1. **§6.1 two-bank KNN + §6.4 failure-aware threshold calibration** — zero training, immediately uses failure data, addresses the structural asymmetry of the current scoring rule. The cheapest real test of the hypothesis that failure data carries usable signal at all.
2. **§6.2 learned metric (linear projection) + two-bank KNN** — minimal-parameter supervised step. Compatible with §6.1 and §6.3.
3. **§6.3 frozen foundation encoder + steps 1–2 on top** — the only listed move that attacks the "WAM dynamics is data-starved" root cause.
4. **§6.5 MIL** — only if §6.2/§6.3 plateau and the diagnosis is "after-GT labels are too dirty."
5. **§6.6 contrastive encoder fine-tune** — last resort.

## 8. Evaluation notes

- Always evaluate on **real-world** and **sim** separately. Sim numbers are saturated; the real bar is real-world AgileX.
- Report both **trajectory-level** (does this rollout fail at all?) and **frame-level** (when does it first fail?) metrics, since downstream uses differ. The existing `FailureBenchmark` already produces both.
- Sanity check: any new head must preserve or improve the **local** (t-SNE-visible) separability observed in `run_20260511_170606_vis`. Linear PCA separability is not a target — the raw latent does not have it and that is fine.

---

*Author: working notes, 2026-05-12. Living document — update as experiments land. Visualization referenced: `checkpoints/lpb_v2/real_world_eval/run_20260511_170606_vis/`.*
