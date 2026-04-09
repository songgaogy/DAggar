# LPB Score DSM: Problem Overview and Background

This document provides a high-level overview of the LPB score discriminator project in `robosuite/discriminator/lpb_score`.

It focuses on:

- what problem is being solved
- why the problem is difficult in this setting
- where data comes from and what it represents
- the conceptual algorithm idea (without code-level details)
- how to interpret current observed behavior and limitations

This file intentionally avoids implementation-level details and prescriptive improvement plans.

---

## 1) What Problem Is Being Solved

### 1.1 Core objective

The system aims to detect manipulation failures from trajectory data in a **multi-task robotic environment**.

At each time step, the detector receives a compact representation of behavior and asks:

> "Does this transition look like normal task execution, or does it look off-manifold?"

The final output is a time series of anomaly scores and a failure/no-failure decision signal.

### 1.2 Why this matters

In real deployment, a policy may continue acting even after task intent is already broken (wrong object, wrong order, irreversible scene state). A detector is needed to:

- identify when behavior deviates from expected task semantics
- provide online or offline safety/monitoring cues
- support automated analysis of large rollout sets

### 1.3 Practical success criterion

A useful detector should do more than classify "good vs bad" at trajectory end. It should also:

- detect failures early
- remain stable across multiple tasks
- avoid many false alarms on normal but diverse behaviors

---

## 2) Background and Motivation

### 2.1 Context: latent-based monitoring

The project builds on latent states extracted from a frozen policy encoder. Instead of operating directly on pixels, the detector works in this latent/action space because it is:

- compact
- policy-aligned
- computationally efficient for large-scale rollout analysis

### 2.2 Why not plain supervised classification

A direct binary classifier usually needs clean failure labels and broad failure coverage. In manipulation, failures are diverse and long-tailed:

- contact mistakes
- wrong object interactions
- delayed consequences
- physically plausible but semantically wrong behavior

This motivates an "energy from normality" perspective rather than strict closed-set classification.

### 2.3 Why score-matching style ideas

The chosen direction models transition plausibility by learning a denoising-based energy signal:

- low energy for transitions near learned normal manifold
- higher energy for transitions that appear inconsistent with learned structure

This is conceptually close to learning a unified world-action consistency model in latent space.

---

## 3) Data Sources and Their Meaning

### 3.1 Data families

The dataset is organized per task into three behavior families:

- **expert**: high-quality demonstrations
- **success_rollout**: policy rollouts that still complete the task
- **fail_rollout**: policy rollouts that do not complete task objectives

### 3.2 Multi-task coverage

The setup includes several manipulation tasks (lift, pick-place variants, stacking). This creates realistic cross-task variation in:

- object interactions
- temporal structure
- motion style
- latent distribution geometry

### 3.3 What each trajectory contributes

Each trajectory contributes:

- a latent sequence (state-like representation over time)
- an action sequence (control signal over time)

The detector reasons on transitions formed from these sequences over a chosen horizon.

### 3.4 Important data ambiguity

Not all "failures" are equally obvious in latent transition space. Some failure rollouts remain locally plausible for many steps before semantic failure becomes visible, which is a central challenge for this project.

---

## 4) Problem Structure: What Must Be Detected

Failure detection in this project is not only about collisions or unstable dynamics. Many important failures are **semantic**:

- the agent interacts with the wrong object
- object order is incorrect
- a subgoal is silently violated while motions remain smooth

This creates a mismatch between:

- **local physical plausibility** (often present in wrong behavior)
- **global task correctness** (which may already be broken)

A robust detector must infer this mismatch from latent transitions alone, which is fundamentally hard.

---

## 5) Conceptual Algorithm Summary (High Level)

### 5.1 Transition-level factorization idea

A transition can be evaluated from three complementary viewpoints:

1. **State familiarity**  
   Is the current latent state itself typical?
2. **Action consistency**  
   Is the chosen action typical under the current state?
3. **Transition dynamics consistency**  
   Is the resulting next state consistent with state + action context?

These viewpoints are combined into a unified anomaly signal per step.

### 5.2 Unified modeling idea

Instead of training fully separate detectors per factor, the method uses one shared model family with routed conditioning. Conceptually, this encourages:

- shared representation across tasks/factors
- parameter efficiency
- consistent energy scale across components

### 5.3 Denoising-energy interpretation

The model is trained to recover clean transition structure from perturbed inputs. The reconstruction quality and uncertainty induce an energy-like score:

- lower score => transition appears normal
- higher score => transition appears atypical

### 5.4 Temporal decision layer

Per-step energies are then aggregated over time to form a trajectory-level decision signal. This separation creates two conceptual layers:

- representation layer: "how abnormal is each step?"
- decision layer: "when is abnormality strong enough to call failure?"

---

## 6) Why This Problem Is Hard

### 6.1 Local plausibility vs global semantics

Many failures are semantically wrong but still locally smooth and physically plausible. A detector based on short transitions may under-react if each local step appears consistent.

### 6.2 Multi-task manifold overlap

Across tasks, latent neighborhoods can overlap in complex ways. A transition that is atypical for one task may still look plausible under a broader shared manifold.

### 6.3 Small-data constraints

With limited trajectories per mode, it is difficult to learn fine-grained semantic boundaries beyond coarse motion plausibility.

### 6.4 Label granularity mismatch

Trajectory-level failure labels do not always indicate the exact onset step. This makes precise temporal calibration difficult and can blur supervision signals.

### 6.5 Distribution diversity inside "normal"

Even successful executions can be diverse in timing, contact dynamics, and sub-strategy. A detector must avoid collapsing normal diversity into false positives.

---

## 7) Interpreting Current Observations

### 7.1 Reported phenomenon

Current experiments suggest performance can be close to latent KNN baselines in some settings.

### 7.2 High-level interpretation

This can happen when learned energy behaves primarily as a smooth distance-like normality score in latent/action space. Under limited semantic separation, such a score may resemble strong nearest-neighbor behavior.

### 7.3 Example pattern: semantic mismatch in stacking

In cases like object-role confusion during stacking (for example, manipulating the wrong block), behavior may remain locally plausible in transition space. The detector can then produce weak or delayed anomaly growth even though task intent is already violated.

### 7.4 What this says about the system

The observed behavior does not necessarily mean the framework is incorrect. It often indicates a tension between:

- what latent transitions encode well (local dynamics)
- what users care about operationally (semantic task correctness)

---

## 8) System-Level Mental Model

A helpful way to reason about the detector is to separate two layers:

### 8.1 Layer A: transition plausibility modeling

Learns whether step-level transitions look normal under learned latent-action structure.

### 8.2 Layer B: temporal alarm policy

Converts noisy step evidence into stable failure alarms through aggregation and thresholding.

When analyzing behavior, both layers matter:

- weak step evidence leads to weak alarms even with aggressive thresholds
- strong step evidence can still be delayed by conservative temporal policies

---

## 9) Terminology and Reading Guide

### 9.1 Useful terms

- **normal manifold**: region in latent/action transition space representing expected behavior
- **energy/anomaly score**: scalar measure of transition atypicality
- **horizon**: number of future action steps included in one transition
- **aggregation**: temporal operation converting step scores to decision scores
- **calibration**: mapping score statistics to thresholds/operating points

### 9.2 How to read project artifacts

- Use trajectory visualizations to understand semantic failure timing.
- Use score curves to understand anomaly growth dynamics.
- Use task-level summaries to distinguish global vs task-specific behavior.

---

## 10) Bottom-Line Summary

This project tackles a hard and important problem: robust failure detection from latent trajectories in multi-task manipulation.

The core idea is to measure transition consistency through a unified denoising-energy framework and convert that signal into temporal failure decisions.

Current observed limitations are consistent with known challenges in latent-based anomaly detection:

- semantic errors can remain locally plausible
- multi-task manifolds can blur boundaries
- small data can constrain semantic separability

As a result, the system may sometimes behave similarly to strong latent-distance baselines, especially on subtle semantic failures.
# LPB Score DSM: End-to-End Technical Summary

This document provides a full technical overview of the LPB score discriminator under `robosuite/discriminator/lpb_score`, including:

- problem definition
- data and feature pipeline
- model architecture and objective
- training and inference behavior
- why the current method may saturate near latent KNN
- concrete upgrade roadmap and experiment plan

The intended reader should be able to understand and improve the method without first reading all source files.

---

## 1) Problem Definition and Scope

### 1.1 Target task

Build an **offline failure detector** for multi-task robot manipulation trajectories, using latent features extracted from a frozen policy encoder.

Given transition tuple:

- `z_t` (current latent)
- `a_{t:t+H-1}` (action chunk of horizon `H`)
- `z_{t+H}` (future latent)

the method outputs a transition-level anomaly energy and trajectory-level failure decision.

### 1.2 Design objective

The detector is not trying to imitate rewards. It tries to estimate how much a transition deviates from a learned normal manifold (expert/success behavior), and then converts that energy to a binary failure alarm via temporal aggregation + threshold calibration.

### 1.3 Why this design

The approach uses a score-matching-style denoising objective to approximate a unified world-action consistency model that can operate across multiple tasks with one shared backbone.

---

## 2) Data Sources and Preprocessing

## 2.1 Raw data types per task

From YAML configs (`config/train.yaml`, `config/eval_dsm_discriminator.yaml`), each task has:

- `expert_dir`
- `success_rollout_dir`
- `fail_rollout_dir`

Tasks include:

- `PandaLift`
- `PandaPickPlaceCan`
- `PandaStack`
- `PickPlaceBread`
- `PickPlaceCereal`
- `PickPlaceMilk`

### 2.2 Split policy

Each task config defines `train / validation / test` trajectory counts for each data type.

Important detail:

- Default train and validation composition for DSM uses only `["expert", "success_rollout"]`.
- Fail rollouts are primarily for evaluation/analysis unless explicitly added to training objective.

### 2.3 Feature extraction and caching

Pipeline (`app/pipeline.py`, `core/dataset.py`):

1. Build split references.
2. Encode frames with `FrozenFlowMultitaskEncoder`.
3. Cache trajectories (`latents[T,D]`, `actions[T,A]`) to `data.cache_dir`.
4. Build transition samples using horizon `H`.

Transition sample in `LatentTransitionDataset`:

- `current_latent = latents[t]`
- `action_sequence = actions[t:t+H]`
- `target_latent = latents[t+H]`
- plus metadata: `task_index`, `data_type_index`, `is_expert`

### 2.4 Normalization statistics

In `app/train.py`, `_compute_expert_normalization_stats` computes:

- `latent_mean`, `latent_var`
- `action_mean`, `action_var`

using **expert training refs only**.

These stats are copied into model buffers and saved in checkpoint payload for deterministic inference consistency.

---

## 3) Mathematical Formulation

### 3.1 Factorization

The method keeps causal-style factorization:

`p(z_t, a, z_{t+H}) = p(z_t) p(a|z_t) p(z_{t+H}|z_t,a)`

implemented as three routed denoising tasks sharing parameters.

### 3.2 Normalized variables

Let:

- `z_t_norm = (z_t - mu_z) / sigma_z`
- `a_norm = (a - mu_a) / sigma_a`
- `z_next_norm = (z_{t+H} - mu_z) / sigma_z`
- `delta_z_norm = z_next_norm - z_t_norm`

where `sigma` is clamped by `std_clamp_min` to avoid division instability.

### 3.3 Noisy training targets

Noise is injected in normalized space:

- `z_noisy = z_t_norm + eps_z`
- `a_noisy = a_norm + eps_a`
- `delta_noisy = delta_z_norm + eps_delta`

with Gaussian noise scale `noise_scale`.

### 3.4 Per-dimension energy

Model predicts `(pred_mean, pred_logvar)` and uses heteroscedastic Gaussian NLL:

`E_dim = 0.5 * ((pred_mean - target)^2 * exp(-pred_logvar) + pred_logvar)`

### 3.5 Per-step transition energy

Mean-reduce each routed branch over its active dimensions:

- `E_state`
- `E_action`
- `E_dyn`

Final step energy:

`u_t = E_state + E_action + E_dyn`

Training loss is batch mean of `u_t`.

---

## 4) Model Architecture in Code

Primary file: `core/model.py`.

### 4.1 Unified routed transformer (`UnifiedConditionedDSM`)

- Shared transformer encoder backbone.
- Learned route token via `task_embedding(3, embed_dim)`.
- Three route IDs:
  - `0` state
  - `1` actor
  - `2` dynamics

Per route, sequence is:

1. task token
2. context token (`context_proj`)
3. target token (`target_proj`)

Only the transformed target token is decoded by:

- `output_mean`
- `output_logvar`

### 4.2 Route-specific context/target design

- **State route**: target=`z_t_norm`, context=`[]`
- **Actor route**: target=`a_norm`, context=`z_t_norm`
- **Dynamics route**: target=`delta_z_norm`, context=`[z_t_norm, a_norm]`

### 4.3 Shared-width padding strategy

Because each route has different target/context widths, code pads them to:

- `target_max_dim = max(latent_dim, action_flat_dim)`
- `context_max_dim = latent_dim + action_flat_dim`

then uses shared linear projections.

This keeps one backbone for all routed factors with minimal architectural branching.

---

## 5) Training System Behavior

Primary files: `app/train.py`, `core/trainer.py`.

### 5.1 Optimization

- optimizer: `AdamW`
- objective: `compute_dsm_loss` from model
- gradient clipping: `grad_clip_norm`
- periodic checkpointing with metadata payload

### 5.2 Sampling policy

`Trainer` supports balanced expert-vs-rollout sampling with `WeightedRandomSampler` and `expert_sampling_ratio`.

Default from config:

- `train_data_types = expert + success_rollout`
- `expert_ratio = 0.5`

This means training intentionally covers expert and policy-success transition spaces.

### 5.3 Current default hyperparameters (important ones)

From `config/train.yaml`:

- `transition_horizon = 1`
- `embed_dim = 512`
- `num_layers = 4`
- `num_heads = 8`
- `ffn_dim = 2048`
- `dropout = 0.0`
- `std_clamp_min = 0.05`
- `noise_scale = 0.08`
- `epochs = 50`

---

## 6) Inference and Detection Logic

Primary file: `core/dsm_discriminator.py`.

### 6.1 Checkpoint load and consistency

`DSMTransitionScorer` loads:

- model weights
- architecture config
- `normalization_stats`
- training horizon

Inference `action_horizon` must match checkpoint horizon.

### 6.2 Step score computation

For each valid time step:

1. build `(z_t, action_chunk, z_{t+H})`
2. run model with `add_noise=False`
3. compute branch NLL energies
4. sum to `step_score = u_t`

### 6.3 Temporal aggregation (`lambda_t`)

Two modes:

- `mean`: prefix or rolling average
- `max`: prefix max or rolling max

`lambda_window_size <= 0` means full prefix accumulation.

### 6.4 Threshold calibration and decision

`fit()` computes:

- global calibration lambda distribution
- per-task calibration lambda distributions

Threshold:

- quantile defined by `delta` (`q = 100*(1-delta/100)`)
- task-specific threshold preferred if available

Prediction at step `t`:

- fail if `lambda_t >= threshold_t`

### 6.5 Adaptive online threshold option

`detect_trajectory` supports optional adaptive delta update:

- adjusts `delta` online based on label/pred mismatch
- clamps to `[delta_min, delta_max]`
- recomputes threshold from calibration distribution

This changes operating point but not representation power.

---

## 7) Why Performance Can Approach Latent KNN

Even though architecture is transformer-based, practical inference energy behaves like a learned distance-like metric on normalized latent/action transitions:

1. deterministic clean-input scoring at test time (`add_noise=False`)
2. NLL with predicted variance acts as adaptive weighting across dimensions
3. expert-based normalization already enforces feature scaling

With limited data and weak semantic separation in latent space, this can perform similarly to strong distance baselines (e.g., latent KNN), especially for local plausibility errors.

---

## 8) Failure Modes and Root-Cause Hypotheses

### 8.1 High-priority failure modes

1. **Semantic mismatch not captured by short horizon**
   - `H=1` can miss long-horizon intent violations.
2. **Task semantics not injected into model forward**
   - dataset includes `task_index`, but model currently does not consume it in routing context.
3. **Broad normal manifold**
   - including success rollout in training can normalize policy-typical but semantically wrong states.
4. **Variance inflation in uncertain dimensions**
   - high `pred_logvar` can reduce anomaly penalty.
5. **Aggregation smoothing**
   - mean + wider window can suppress short critical spikes.

### 8.2 PandaStack wrong-object example

Observed behavior: robot may manipulate the wrong block (semantic error) while local transitions remain dynamically plausible.

Why detector may not fire early:

- latent may encode motion plausibility stronger than object-role semantics
- one-step delta remains near manifold
- aggregated lambda smooths brief deviations

---


## 9) Bottom-Line Summary

The current LPB score DSM is a clean and coherent unified denoising energy framework, but its default setup (short horizon, no explicit task conditioning, broad normal training manifold, smoothing-heavy aggregation) can limit semantic anomaly detection and make performance close to latent KNN in difficult cases.

The highest ROI path is:

1. run the compact 8-run matrix,
2. add task conditioning,
3. add a simple ranking/negative objective.

This sequence usually gives the fastest evidence on whether the bottleneck is calibration-only or fundamentally representational.
