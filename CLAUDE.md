# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository shape

This is a fork of ARISE-Initiative/robosuite (v1.5.2, MuJoCo-based robot simulation) extended for interventional imitation-learning + failure-detection research. Upstream robosuite lives under `robosuite/{environments,models,controllers,robots,renderers,wrappers,scripts,demos}` and is largely untouched. The research code is layered on top in sibling packages:

- `robosuite/policy/` — action policies
  - `flow_multi/` is the **active multitask flow-matching policy** (ResNet18 + CLIP text + AdaLN-FiLM fusion + UNet-1D head). Checkpoints live under `checkpoints/multitask_6/policy/flow-20/flow_multi_ep*.pt` and are the init-checkpoint for flow-DAgger.
  - `flow_unet/` and `flow_v0/` are earlier single-task variants still used for baselines and data generation.
- `robosuite/pipeline/` — Hydra-configured online/interventional training pipelines. Algorithms (`hil-serl`, `hg-dagger`, `flow-dagger`) and runtime components (policies, online discriminators) are registered in `pipeline/factory.py` and built from `pipeline/config/*.yaml`. The main entrypoints are `pipeline/train_{flow_dagger,hg_dagger,hil_serl}.py` and `pipeline/eval_flow_dagger.py`. Shared runtime in `pipeline/envs/` wraps robosuite envs; shared utilities (async checkpoint/buffer writers, viewer runtime, jsonl event logger, FPS/interval gates) are in `pipeline/utils/`.
- `robosuite/discriminator/` — failure / OOD detectors over latent transitions from the frozen flow encoder. Several variants coexist:
  - `lpb_score/` — **current SσDC (Single-σ Diffusion Classifier)** detector. Task-conditioned latent DSM trained with fixed σ=0.08 on (state, dynamics) factors; action is observed conditioning only. Read `lpb_score/summary.md` before touching this. Key files: `core/model.py` (`UnifiedConditionedDSM`, `DSMModel`), `core/dsm_discriminator.py`, `core/trainer.py` (AdamW + EMA + per-epoch `class_conditional_diff` collapse monitor — watch for `WARNING[collapse]`).
  - `lpb_new/` — previous kNN-based LPB discriminator (still referenced by `pipeline/factory.py` as the `lpb_new` online discriminator).
  - `dyn_bce/` / `bce/` — BCE-style discriminators. `FrozenFlowMultitaskEncoder` in `dyn_bce/modules/flow_encoder.py` is the shared encoder used by all latent-based detectors.
  - `visual_ebd/` — visual embedding-based analysis.
- `robosuite/RL/` — vendored CleanRL single-file algorithms + our SAC/DrQ-v2 Lift-image variants. Only `RL/discriminator/` and `RL/models/resnet18-f37072fd.pth` are imported by the policy/discriminator code; the rest is benchmarking scaffolding.
- `robosuite/run/` — the canonical shell wrappers that launch training/eval/collection. **Prefer editing these over ad-hoc command lines** — they encode the paths, GPU assignment, and hyperparameters people actually use.

## Environment & tooling

- Python is run via the conda env `daggar`. Scripts hard-code `/home/dodo/miniconda3/envs/daggar/bin/python` (see `lpb_score/scripts/train_lpb_score_dsm.sh`). When invoking Python yourself, use that interpreter.
- The repo is editable-installed (`pip install -e .`, declared in `setup.py` as `robosuite==1.5.2`).
- Formatting: `black` (line 120) and `isort` (black profile); excludes `robosuite/models/assets` and `robosuite/controllers/config`. `pre-commit install` sets up both as hooks.
- Tests: `python -m pytest` from repo root runs upstream robosuite tests under `tests/`. There is no test coverage for the pipeline/, policy/, or discriminator/ research code.

## Data and checkpoint layout

Per task under `./data/<TASK>/`:
- `expert/` — scripted success demos (HDF5) — source of truth for BC/flow training
- `success_rollout/` — replay-verified successes
- `fail_rollout/` — failure rollouts (`traj_type = 1` in the SσDC dataset)
- `unlabled/` — unlabeled rollouts for discriminator training
- `expert_recover/` — for flow_unet training

Task names in use: `PandaLift`, `PandaPickPlaceCan`, `PandaStack`, `PickPlaceBread`, `PickPlaceCereal`, `PickPlaceMilk`. The flow-multi eval benchmark set is `[PickPlaceCan, PickPlaceBread, PickPlaceCereal, PickPlaceMilk]`.

Checkpoints under `checkpoints/`:
- `multitask_6/policy/flow-20/flow_multi_ep*.pt` — frozen multitask flow policy (used as both the init checkpoint for flow-DAgger and the encoder for all latent discriminators).
- `multitask_6/lpb_dipole-new-v*/` — SσDC training outputs (`lpb_score`).
- `<task>/flow/BC_warmup/`, `<task>/discriminator/{bce,lpb_knn,lpb_dynamics}/` — per-task baselines.

Training runs write to `outputs/<pipeline_name>/<run>/` with `console.log`, `metrics_runtime.jsonl`, `buffers/{online_chunks,demo_chunks}/`, `checkpoints/{step_*.pt,latest.pt}`, and a resolved Hydra config.

## Common commands

All training/eval is orchestrated via shell wrappers in `robosuite/run/` and `robosuite/discriminator/lpb_score/scripts/`. Invoke them directly; they set `CUDA_VISIBLE_DEVICES` and resolve paths relative to the repo root.

- Train multitask flow policy: `bash robosuite/run/train_flow.sh`
- Train flow-DAgger online (interventional): `python -m robosuite.pipeline.train_flow_dagger` (config: `robosuite/pipeline/config/train_flow_dagger.yaml`, Hydra overrides OK on CLI; reads `runtime.init_checkpoint` as the frozen flow policy).
- Train HG-DAgger / HIL-SERL: `python -m robosuite.pipeline.train_hg_dagger` / `train_hil_serl` with the matching yaml.
- Train SσDC discriminator: `bash robosuite/discriminator/lpb_score/scripts/train_lpb_score_dsm.sh` (or `python -m robosuite.discriminator.lpb_score.train` with Hydra overrides).
- Benchmark SσDC: `bash robosuite/discriminator/lpb_score/scripts/run_lpb_score_benchmark.sh`. Weight overrides via env: `ALPHA_STATE`, `ALPHA_DYNAMICS`, `BETA_STATE`, `BETA_DYNAMICS`, `LAMBDA_MODE`, `LAMBDA_WINDOW_SIZE`, `DELTA`. Defaults are β-only (R5 recipe).
- Collect human demos: `bash robosuite/run/collect_human_data.sh` (wraps `robosuite/scripts/collect_human_demonstrations.py`; uses spacemouse).
- Upstream tests: `python -m pytest` (robosuite env/controller/gripper/renderer smoke tests). To run a single test: `python -m pytest tests/test_environments/test_all_environments.py -k lift`.

## Architectural notes that span files

- **Flow-DAgger data flow** (`pipeline/train_flow_dagger.py`): `flow_multi` init checkpoint supplies `camera_names`, `task_prompt_map`, `model_cfg`, action-horizon mean/std, and per-task env metadata (`resolve_flow_task_metadata`). The main env renders RGB observations at `env.img_height × env.img_width`, center-cropped and resized before the policy. A **decoupled viewer env** (second robosuite instance) can be spawned for display while the training env uses offscreen rendering (`runtime.viewer_enabled` + online updates → `decoupled_viewer_enabled`). Async learner updates run on a background thread (`trainer.start_async_worker()`). Transitions stream to `buffers/online_chunks` and (on intervention) `buffers/demo_chunks` via `AsyncTransitionChunkWriter`; checkpoints go through `AsyncCheckpointWriter`. Intervention comes from a spacemouse via `RobosuiteInterventionRuntime`.
- **SσDC detector** (`discriminator/lpb_score/`): per transition `τ = (z_t, a, z_{t+H})` with class `c ∈ {0:success, 1:failure}`, factorize `p(τ|c) = p(z_t|c)·p(z_{t+H}|z_t,a,c)`. A shared DiT backbone (`UnifiedConditionedDSM`) with two routed target encoders + heads (`state`, `dynamics`) and one `action_context_encoder` (dynamics branch only) computes per-factor energies at inference with `add_noise=False`. The step score is `u(x) = Σ_b [α_b·z(E_b⁺) + β_b·(z(E_b⁺) − z(E_b⁻))]` over success-calibrated per-task z-scores, aggregated into λ by rolling mean/max and thresholded by `delta` percentile. Action is **never** denoised — dropping the old `p(a|z_t,c)` factor was the point of the SσDC refactor, and the pre-refactor `lpb_dipole-new` checkpoints are incompatible. The `[collapse]` epoch log is the canary for class-signal collapse; below `1e-4` for 3 consecutive epochs blocks commits.
- **Registry pattern**: `pipeline/factory.py` registers policies (`dummy`, `flow_multi`), online discriminators (`none`, `lpb_new`, `bce`), and algorithms (`hil-serl`, `hg-dagger`, `flow-dagger`) via decorators; they are selected by string `type` in Hydra config. Add new ones with `@register_{policy,discriminator,algorithm}`.
- **Offline demo cache**: `pipeline/utils.load_demo_paths` caches parsed HDF5 demos under `outputs/<run>/../_demo_cache/` keyed by `(img_h, img_w, camera_names, proprio_keys)`. Changing camera/image parameters invalidates the cache.

## Conventions specific to this fork

- Imports across the research layer assume absolute module paths (`from robosuite.pipeline...`, `from robosuite.discriminator...`, `from robosuite.policy.flow_multi...`). Never add relative imports between these packages.
- Hydra runs with `hydra.run.dir=.` and `output_subdir=null` at the top level; training scripts build their own dated run directories (see `resolve_run_directory` in `train_flow_dagger.py`) so Hydra does not litter `.hydra/` into cwd.
- Shell wrappers compute `ROOT_DIR` relative to their own path and use `${CUDA_VISIBLE_DEVICES:-0}`. Mirror that pattern for new wrappers instead of hard-coding absolute `/home/dodo/...` paths (several older scripts still do — prefer the newer `lpb_score/scripts/*.sh` style).
- W&B: set via `logging.use_wandb`, `logging.wandb_mode=offline` by default; entity is `songgao-personal`. Runs sync later from `wandb/offline-run-*`.
