# robosuite.pipeline

This pipeline tree is trimmed for DIPOLE training and evaluation.

## Entry Points

Use the helper scripts from the repo root:

```bash
# Online DIPOLE training with optional GUI/intervention.
bash robosuite/pipeline/scripts/train_dipole.sh

# Headless checkpoint evaluation and optional omega sweep.
CHECKPOINT=outputs/DIPOLE/<run>/checkpoints/latest.pt \
bash robosuite/pipeline/scripts/eval_dipole.sh
```

Direct module entry points are also available:

```bash
/home/dodo/miniconda3/envs/dagger/bin/python -m robosuite.pipeline.train_dipole
/home/dodo/miniconda3/envs/dagger/bin/python -m robosuite.pipeline.eval_dipole --help
```

The scripts assume the repo root is `$HOME/Documents/DAggar/robosuite` by
default. Override `ROOT_DIR` when running from another checkout location.

## Retained Structure

- `algorithms/dipole/`: DIPOLE agent, trainer, replay buffer, flow policy, and LPB v2 BCE `G` provider.
- `algorithms/flow_dagger/`: internal dataclasses, replay-buffer base, and model/image helpers still imported by DIPOLE.
- `config/train_dipole.yaml`: Hydra config for DIPOLE.
- `envs/`, `utils/`, `common/`: robosuite runtime, IO, logging, checkpointing, and shared types used by DIPOLE.
- `utils/diagnose_dipole_divergence.py`: static DIPOLE checkpoint diagnostic.

Non-DIPOLE standalone training/evaluation entry points were removed from this
pipeline surface. DIPOLE still supports loading a flow-policy base checkpoint
through `runtime.init_checkpoint`.

## GUI And Runtime Notes

Interactive DIPOLE runs use these flags:

- `runtime.interactive=true`
- `runtime.viewer_enabled=true`
- `intervention.enabled=true`

When `INTERACTIVE=true`, `scripts/train_dipole.sh` exports `MUJOCO_GL=glfw`
and requires a non-empty `$DISPLAY`. For headless runs, use:

```bash
INTERACTIVE=false VIEWER_ENABLED=false INTERVENTION_ENABLED=false \
bash robosuite/pipeline/scripts/train_dipole.sh
```

Headless mode exports `MUJOCO_GL=egl`.

## Logging

DIPOLE defaults to WandB offline logging:

```bash
WANDB_MODE=offline
WANDB_ENTITY=songgao-personal
```

Run artifacts are written under `outputs/DIPOLE` unless `logging.output_root`
is overridden.
