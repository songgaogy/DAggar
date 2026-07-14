# LPB-Original Debug Session

Date: 2026-04-25
Branch: `discriminator-baseline`
Working dir: `/home/dodo/Documents/DAggar/robosuite_dipole-rl_v3`

## Context

Debugging session for `robosuite/discriminator/lpb_original/` after the initial port
(documented in `lpb_original_port.md`). Two user constraints carried into this session:

1. **Data source**: pre-computed cache (`data/.lpb_score_preprocessed_cache/<task>/*.npz`)
   via `env=preprocessed` (i.e. `PreprocessedCacheDynamicsModelDataset`).
2. **No policy checkpoint**: if `policy_ckpt_path` is null, `ResNetEncoder` falls back
   to torchvision ResNet18 ImageNet weights. Already handled in `resnet_encoder.py`.

## Cache Structure

```
data/.lpb_score_preprocessed_cache/
├── PandaLift/          *.npz  (T, V=3, 3, 128, 128) images_chw + (T,14) proprio + (T,7) actions
├── PandaPickPlaceCan/
├── PandaStack/
├── PickPlaceBread/
├── PickPlaceCereal/
└── PickPlaceMilk/
```

Each `.npz` stores `images_chw (T, V, 3, H, W)`, `proprio (T, 14)`, `actions (T, 7)`.
`V=3` views; camera index 0 = agentview. `proprio_dim=14` (not 9!).

## Bugs Found & Fixed

### Bug 1 (Critical) — `train.py`: `tasks` not forwarded to `PreprocessedCacheDynamicsModelDataset`

**File**: `robosuite/discriminator/lpb_original/train.py`  
**Function**: `_instantiate_dataset`

**Symptom**: `env=preprocessed` yaml defines
`tasks: [PandaLift, PandaPickPlaceCan, PandaStack, PickPlaceBread, PickPlaceCereal, PickPlaceMilk]`
but `_instantiate_dataset` never forwarded `tasks` or `cache_root` in kwargs.
`PreprocessedCacheDynamicsModelDataset` defaulted to `tasks=("PickPlaceBread",)` only →
**5.5× fewer training samples** (158k vs 881k).

**Fix**: Added to `_instantiate_dataset`:
```python
if "cache_root" in cfg.env and cfg.env.cache_root is not None:
    kwargs["cache_root"] = hydra.utils.to_absolute_path(str(cfg.env.cache_root))
if "tasks" in cfg.env and cfg.env.tasks is not None:
    kwargs["tasks"] = list(cfg.env.tasks)
```

---

### Bug 2 (Critical) — `plan.py`: `prior_in_chans` hardcoded to 9

**Files**: `dyn_model/plan.py` + `train.py`  
**Function**: `plan.load_model` (called by `LPBOriginalEncoder.__init__`)

**Symptom**: `plan.load_model` reconstructs the `proprio_encoder` using a hardcoded
`prior_in_chans = 9` (with path-string heuristics for libero/transport/pusht).
The preprocessed cache data has `proprio_dim=14`. At checkpoint load time:
`ProprioceptiveEmbedding.patch_embed.weight` expected shape `[32, 9, 1]`
but checkpoint contained `[32, 14, 1]` → **RuntimeError: size mismatch**.

**Root cause**: `plan.py` is upstream code that infers dims from dataset path names;
our HDF5/preprocessed paths don't contain 'libero'/'transport'/'pusht', so `prior_in_chans`
stays at the hardcoded default of 9.

**Fix 1** — `train.py` (`main()`): save actual dims into cfg **before** writing `hydra.yaml`:
```python
with open_dict(cfg):
    ...
    cfg.prior_in_chans = int(train_ds.proprio_dim)   # e.g. 14
    cfg.action_dim_per_step = int(cfg.env.action_dim) # e.g. 7
OmegaConf.save(cfg, out_dir / "hydra.yaml", resolve=True)
```

**Fix 2** — `dyn_model/plan.py` (`load_model`): read saved values if present,
overriding the path-name heuristics:
```python
if getattr(train_cfg, 'prior_in_chans', None) is not None:
    prior_in_chans = int(train_cfg.prior_in_chans)
if getattr(train_cfg, 'action_dim_per_step', None) is not None:
    action_dim = int(train_cfg.action_dim_per_step)
```

Backwards-compatible: old checkpoints without these keys fall through to the original logic.

---

### Side-effect fix — `train.py` `main()` restructure

The original code instantiated datasets **after** saving `hydra.yaml`, making it
impossible to embed runtime dims into the saved config. The restructured order is:

1. Instantiate `train_ds`, `valid_ds` (needed to know `proprio_dim`)
2. Add `prior_in_chans` / `action_dim_per_step` to cfg
3. Save `hydra.yaml`
4. Build model, normalizer, run training loop

## Verifications

- `PreprocessedCacheDynamicsModelDataset(tasks=all_6_tasks)` → 881,279 samples ✓
- `PreprocessedCacheDynamicsModelDataset(tasks=['PickPlaceBread'])` → 158,826 samples ✓
- `plan.py` override logic: `prior_in_chans=14, action_dim=7` for preprocessed config ✓
- Basic module imports in dagger env ✓

## Files Modified

| File | Change |
|---|---|
| `robosuite/discriminator/lpb_original/train.py` | `_instantiate_dataset`: add `tasks`/`cache_root` forwarding; `main()`: save `prior_in_chans`/`action_dim_per_step` to cfg before hydra.yaml write; remove duplicate dataset instantiation |
| `robosuite/discriminator/lpb_original/dyn_model/plan.py` | `load_model`: read `prior_in_chans` and `action_dim_per_step` from saved config when present |

## Not Yet Run

- Full end-to-end training with `env=preprocessed` (requires GPU, takes hours).
- Checkpoint loading via `LPBOriginalEncoder` on a real trained checkpoint.
- KNN benchmark run.

## How to Train (preprocessed cache path)

```bash
cd /path/to/robosuite
conda activate dagger
python -m robosuite.discriminator.lpb_original.train env=preprocessed \
  hydra.run.dir=checkpoints/lpb_original/dynamics/$(date +%Y.%m.%d/%H.%M.%S)_preprocessed
```

No `policy_ckpt_path` needed — falls back to ImageNet ResNet18.
