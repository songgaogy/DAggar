# Subagent Prompt 07 — End-to-End Verification

You are running the full DIPOLE-RL pipeline end-to-end and confirming
correctness. No new files are created. You will write a small smoke-test
script and report results.

## Background

Once prompts 01–06 are merged, we need to confirm:
1. Every skeleton was filled in.
2. The encoder is shared across the three downstream modules.
3. Warmup runs to completion.
4. Online training produces sane metrics on a short headless run.
5. The legacy `g_mode=bce_frozen` regression path still works.

## Required reading

- `robosuite/pipeline/docs/DIPOLE_RL.md` — §7 (verification).
- Prompts 01–06.

## Deliverables

### Step 1 — Static check

Run from repo root:

```bash
find robosuite/pipeline/algorithms/q_learning \
     robosuite/pipeline/algorithms/discriminator \
     robosuite/pipeline/algorithms/dipole/advantage_g_provider.py \
     robosuite/pipeline/train_dipole_rl.py \
     -name "*.py" -print0 | xargs -0 python -m py_compile
```

Any failure: report file:line.

### Step 2 — Encoder sharing identity

Write `robosuite/pipeline/algorithms/discriminator/tests/test_encoder_sharing.py`
that builds an `IQLLearner` + `OnlineBCEDiscriminator` + `AdvantageGProvider`
with a single `SharedFrozenEncoder` instance and asserts:

- The `id()` of the encoder stored in each downstream module matches.
- `torch.cuda.memory_allocated('cuda:1')` AFTER building the three modules
  is within 10% of the memory allocated by the encoder alone.

### Step 3 — Warmup smoke

```bash
/home/dodo/miniconda3/envs/dagger/bin/python \
  -m robosuite.pipeline.algorithms.q_learning.warmup \
  --value-steps 200 --full-steps 200 \
  --demo-h5 <pick small HDF5 in repo> \
  --output /tmp/iql_warmup_smoke.pt --device cuda:1
```

Assert: exit code 0, `iql_warmup_smoke.pt` size > 1 MB, log shows v_loss
decreasing across the first 200 steps.

### Step 4 — Headless integrated smoke

Modify `robosuite/pipeline/scripts/train_dipole_rl.sh` invocation to run
with:

```bash
INTERACTIVE=false VIEWER_ENABLED=false INTERVENTION_ENABLED=false \
INIT_CHECKPOINT=<flow-dagger ckpt path> \
LPB_CKPT=checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth \
IQL_WARMUP_CKPT=/tmp/iql_warmup_smoke.pt \
ALPHA=1.0 BETA=0.5 G_MODE=advantage \
runtime.max_steps=500 \
bash robosuite/pipeline/scripts/train_dipole_rl.sh
```

After it exits, parse the wandb offline run dir (default `outputs/DIPOLE/
<run>/wandb/latest-run/files/wandb-summary.json`) and assert:

- Metrics keys present: `flow_loss`, `q_loss`, `v_loss`, `disc_loss`,
  `G_mean`, `advantage_mean`, `disc_reward_mean`.
- No value is NaN or Inf.
- `frac_w_pos_saturated_high + frac_w_pos_saturated_low < 0.5` (sanity
  on G dynamic range).

### Step 5 — GPU layout

After Step 4 finishes (or during, if possible), run `nvidia-smi
--query-compute-apps=gpu_uuid,pid,used_memory --format=csv` once and
confirm there is at least one process on each of `cuda:0` and `cuda:1`,
each using < 20 GB. Report the output verbatim.

### Step 6 — Regression on legacy path

Re-run with `G_MODE=bce_frozen` and `algorithm.q_learning.enabled=false`,
`algorithm.discriminator.online_train=false`:

```bash
G_MODE=bce_frozen INIT_CHECKPOINT=<same ckpt> \
LPB_CKPT=checkpoints/lpb_v2/robosuite_ckpt/bce_head.pth \
runtime.max_steps=200 \
algorithm.q_learning.enabled=false \
algorithm.discriminator.online_train=false \
bash robosuite/pipeline/scripts/train_dipole_rl.sh
```

Assert: matches a baseline `train_dipole.sh` run with the same flags
within metric noise (compare `flow_loss` curves visually; log the
divergence). Bit-exactness is not required.

## Done criteria

A single markdown report file `outputs/DIPOLE-RL_verification.md` (this
is the ONLY new file you may create) with a pass/fail line for each of
Steps 1–6, plus the captured `nvidia-smi` output and any error tracebacks.

## Dependency

Depends on prompts **01**, **02**, **03**, **04**, **05**, **06**.
