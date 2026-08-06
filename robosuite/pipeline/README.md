# HIL-SERL Baseline

This package contains the single supported online learning method: HIL-SERL. It is a PyTorch/CUDA port of `tmp/hil-serl` for the robosuite simulator.

## Algorithm

- every executed transition enters the online replay buffer;
- human intervention replaces the policy action and is also copied to the demonstration replay buffer;
- learner batches contain 50% online and 50% demonstration/intervention data;
- one learner step performs `cta_ratio - 1` critic-only updates followed by one critic, grasp critic, actor, and temperature update;
- the actor uses continuous SAC for `action[:-1]` and a three-way grasp critic for the final gripper action;
- policy observations contain only the configured RGB camera images and exclude simulator object state and proprioception;
- robosuite `is_success` supplies the binary `0/1` reward;
- time-limit truncation preserves value bootstrapping;
- a background learner trains continuously after replay warmup and publishes an inference snapshot every 50 learner steps.

Tensor computation is CUDA-only. The pipeline fails instead of falling back to CPU when either configured CUDA device is unavailable.

## Configuration

Hydra composes `config/overall.yaml` with `config/task/PickPlaceCereal.yaml`. The task config is the only supported task in this baseline.

Expert demonstrations are loaded from:

```text
data/PickPlaceCereal/expert
```

The baseline samples 20 trajectories without replacement from the complete expert dataset using the fixed data-selection
seed `42`. The selected source file and demonstration IDs are stored in `demo_selection.json` for every run. Training
seeds do not change this fixed demonstration subset.

Outputs are written below:

```text
outputs/baseline/hil-serl/PickPlaceCereal/<run_name>/
```

Each run contains the resolved config, metadata, TensorBoard events, JSONL and console logs, checkpoints, buffer chunks, and demonstration cache.

## Commands

Train:

```bash
bash robosuite/pipeline/scripts/train.sh
```

Use one GPU for both learner and inference:

```bash
bash robosuite/pipeline/scripts/train.sh \
  runtime.learner_device=cuda:0 \
  runtime.inference_device=cuda:0
```

Evaluate with stochastic SAC sampling:

```bash
CHECKPOINT=/absolute/path/to/latest.pt \
bash robosuite/pipeline/scripts/eval.sh
```

Set `evaluation.deterministic=true` for policy-mean diagnostics.

Visualize Q/V values on a successful rollout:

```bash
CHECKPOINT=/absolute/path/to/latest.pt \
bash robosuite/pipeline/scripts/visualize_q.sh
```

## Package layout

- `src/hil_serl/` contains the agent, trainer, SAC networks, replay, and Q/V diagnostics.
- `src/environment/` contains the robosuite adapter and SpaceMouse intervention runtime.
- `src/data/` contains transition persistence and demonstration loading.
- `utils/` contains shared tensor, runtime, and logging helpers.
- `train.py`, `eval.py`, and `visualize_q.py` are the public command-line entry points.

Checkpoints created before this package layout that contain replay-buffer objects are not supported. Model-only state dictionaries remain portable.
