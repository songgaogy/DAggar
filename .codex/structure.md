You are working inside my local repository and must implement a **PyTorch-only, algorithm-only HIL-SERL** module that fits my existing project structure.

## Repository context

My current directory structure is:

```text
pipeline/
├── algorithms
│   ├── hg_dagger
│   └── hil-serl
├── base.py
├── collect_human_intervention.py
├── config
│   └── collect_human_intervention.yaml
├── factory.py
├── __init__.py
├── scripts
│   └── intervention_rollout.sh
└── utils
```

You must implement the HIL-SERL algorithm under this repository, but **only the algorithmic part**.
Do **not** implement any real robot / hardware / ROS / SpaceMouse / camera server / agentlace networking code.


## First: inspect the local codebase before coding
Before writing code, read these local files carefully and adapt to their conventions:

* `pipeline/base.py`
* `pipeline/factory.py`
* `pipeline/algorithms/hg_dagger/*`
* any existing utilities used by algorithm classes
* any config / rollout / intervention code already present in this repo

Your implementation must match the project’s style and interfaces instead of introducing a disconnected standalone package.

## Second: inspect original official implementation of HIL-SERL
please read `./hil-serl` folder, this is official implementation of hil-serl. It uses `JAX` rather than pytorch, and the repo contains many hardware code.

## What I want you to implement
Implement a **Torch version of HIL-SERL RL core**, distilled from the official HIL-SERL paper/repo, with the following meaning:

### HIL-SERL essentials to preserve

1. **Base RL algorithm**
   * Off-policy actor-critic in the SAC / RLPD family.
   * Pixel + proprio input support.
   * Continuous-action policy as the main path.

2. **Two-buffer training**
   * `online_buffer`: stores all online executed transitions.
   * `demo_buffer` (or `demo_intervention_buffer`): stores:

     * offline demonstrations
     * online intervention transitions
   * Training samples **50% from online buffer and 50% from demo/intervention buffer**.

3. **Human intervention semantics**

   * During rollout, an intervention means the executed action is the human override action.
   * Every executed transition always goes to `online_buffer`.
   * If the transition was executed under intervention, it must also go to `demo_buffer`.
   * This is the most important algorithmic distinction from vanilla demo-bootstrapped SAC.

4. **Sparse success reward interface**

   * In the original HIL-SERL pipeline, reward usually comes from a binary reward classifier.
   * Here, do **not** implement hardware data collection.
   * Instead, design a clean interface so reward can come from:

     * environment reward
     * external reward model / classifier callback
     * precomputed reward field in the batch

5. **Critic-heavy update schedule**

   * Preserve the RLPD-style idea of multiple critic-heavy updates per environment interaction.
   * Implement a configurable `cta_ratio` / update-to-data style loop.
   * Use the pattern:

     * several critic-only updates
     * then one full update (critic + actor + alpha)

6. **Pretrained vision encoder friendly**

   * Design the code so image encoder can be:

     * simple CNN
     * torchvision ResNet backbone
     * frozen pretrained encoder + small projection head
   * Do not hard-code robot-specific image assumptions.

7. **Optional future extension**

   * Structure the code so a hybrid discrete gripper critic (DQN-style grasp head) can be added later.
   * But do **not** block the main implementation on this.
   * First deliver a robust continuous-action HIL-SERL core.

---

## Explicitly out of scope
Do **not** implement these parts:
* ROS / real robot control
* camera drivers
* SpaceMouse input
* agentlace distributed networking
* Franka-specific wrappers
* reward-classifier data collection scripts
* hardware-specific safety controllers

You may create interfaces / hooks for these, but no hardware integration.

## Important repository constraint: folder naming
The existing folder is named:

```text
pipeline/algorithms/hil-serl
```

This is not a good Python import name because of the hyphen.

Handle this cleanly. Use one of these approaches:

* preferred: create an importable package such as `pipeline/algorithms/hil_serl/` and register the algorithm key `"hil-serl"` in `factory.py`
* or keep filesystem compatibility while ensuring Python imports use a valid module path

Do **not** leave the project in a broken import state.


## Deliverables
Implement the algorithm with clean modular code. I want something maintainable, not a giant single file.

At minimum, create or modify modules along these lines:

```text
pipeline/algorithms/hil_serl/
    __init__.py
    agent.py
    sac.py
    buffers.py
    encoders.py
    trainer.py
    types.py
    utils.py
```

Adjust names if needed to match the existing repo style.

Also update integration points such as:

* `pipeline/factory.py`
* possibly `pipeline/base.py` subclasses / registry hooks
* minimal config support if this repo already uses configs

---

## Required algorithm design

### 1. Transition / batch schema

Define a clear transition schema. At minimum each transition should support:

* `obs`
* `action`
* `reward`
* `next_obs`
* `done`
* `is_intervention`
* optional `info`
* optional `reward_source`
* optional `demo_source`

If the current codebase already has a transition abstraction, reuse it.

### 2. Buffers

Implement two replay buffers:

#### Online buffer

* all online executed transitions

#### Demo / intervention buffer

* offline demos inserted at initialization
* intervention transitions appended during online training

Requirements:

* sample mini-batches as tensors
* support images + proprio
* support save/load for checkpoint resume
* robust shape checks

### 3. Agent API

Expose a clean class with methods like:

* `select_action(obs, deterministic=False)`
* `store_online_transition(...)`
* `store_demo_transition(...)`
* `update(...)`
* `save_checkpoint(...)`
* `load_checkpoint(...)`

If the repo already has a base class, conform to it.

### 4. HIL-SERL trainer logic

Implement an in-process trainer that mirrors HIL-SERL logic without networking:

* initialize demo buffer from provided demonstrations
* interact with env / rollout source
* detect intervention flag per step
* route transitions to buffers using the correct semantics
* once warmup is satisfied, run updates
* each update step:

  * sample half-batch from online buffer
  * sample half-batch from demo/intervention buffer
  * concatenate
  * perform critic-heavy update schedule

### 5. SAC / RLPD core

Implement:

* actor
* critic ensemble (at least twin Q)
* target critic
* entropy temperature alpha
* critic loss
* actor loss
* alpha loss
* target soft updates

Use sane defaults similar in spirit to HIL-SERL:

* twin critic
* discount around 0.97
* layer norm in MLPs if appropriate
* image augmentation hook
* configurable entropy tuning

### 6. Visual encoder path

Support observations containing:

* image-only
* proprio-only
* image + proprio

Design:

* image encoder output + proprio concatenation
* projection trunk before actor / critic MLPs

Keep it generic. No robot-specific fields hard-coded beyond what the repo already standardizes.

### 7. Augmentation

Add optional image augmentation in training, e.g. random crop for pixel observations.

Make it configurable and only active for image observations.

### 8. Checkpointing

Checkpoint should include:

* actor / critic / target / alpha optimizer states
* replay buffers
* demo/intervention buffer
* training step counters
* config snapshot if the project already supports it

### 9. Logging / metrics

Track at least:

* actor loss
* critic loss
* alpha / entropy
* mean reward in batch
* online buffer size
* demo buffer size
* intervention rate
* number of intervention transitions added
* update count

Do not introduce external logging dependencies unless the repo already uses them.


## Practical defaults to implement
Use sensible defaults inspired by the official implementation, but adapted to Torch:

* `batch_size = 256`
* `discount = 0.97`
* `training_starts = 100`
* `cta_ratio = 2`
* `steps_per_update = 50`
* twin Q critic
* auto temperature tuning
* optional frozen pretrained visual backbone
* optional random crop augmentation

These should be configurable, not hard-coded.


## Important semantic details

Please preserve these HIL-SERL semantics carefully:

### Buffer routing

* executed online transition:

  * always -> `online_buffer`
* executed online transition with human intervention:

  * also -> `demo_buffer`
* offline demos:

  * preload into `demo_buffer`

### Batch composition

* each training batch should be:

  * half online
  * half demo/intervention

### Algorithm identity

This should feel like:

* **demo + intervention bootstrapped SAC / RLPD**
  not like:
* BC
* DAgger
* offline RL only
* vanilla SAC with demos dumped into one buffer and forgotten

The intervention-aware dual-buffer design is the algorithmic core.

