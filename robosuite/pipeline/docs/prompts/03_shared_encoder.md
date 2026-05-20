# Subagent Prompt 03 — Shared Frozen LPB v2 Encoder

You are implementing `robosuite/pipeline/algorithms/discriminator/encoder.py`.
The skeleton already exists; **fill in the `NotImplementedError` bodies**
and refactor the existing `g_provider.py` to consume your new class
instead of building its own encoder.

## Background

DIPOLE-RL needs a SINGLE frozen LPB v2 encoder shared across three
modules (DIPOLE G provider, IQL learner, OnlineBCEDiscriminator) so the
encoder weights live in cuda:1 memory exactly once. Today's
`LPBV2GProvider.__init__` (g_provider.py:84-151) builds and owns its own
encoder — that has to move out.

The encoder must be wrapped to expose a degraded `f(o, s)` interface:
`encode_batch(images_per_view, proprio, actions := tile(proprio))`. Q
networks re-introduce the action by concatenation, so the encoder side is
action-independent.

## Required reading

- `robosuite/pipeline/docs/DIPOLE_RL.md` — full architecture; especially
  §2 (shared encoder) and §4 (cross-module API).
- `robosuite/pipeline/algorithms/dipole/g_provider.py:69-303` — current
  monolithic encoder + BCE provider; extract the encoder portion.
- `robosuite/discriminator/lpb_v2/detectors/single_bank_knn.py:106-399` —
  the underlying `LPBV2Encoder` you wrap. `encode_batch` lives at lines
  327-399.
- `robosuite/discriminator/lpb_v2/detectors/bce.py:13-25` — BCE ckpt schema
  (you read `feature_source`, `transformer_layer`, `model_ckpt` from it).
- `robosuite/pipeline/algorithms/discriminator/encoder.py` — skeleton you
  must fill in.

## Deliverables

1. Fill in `SharedFrozenEncoder.__init__`: load `bce_ckpt_path`, extract
   the upstream `model_ckpt` + `feature_source` + `transformer_layer`,
   construct an `LPBV2Encoder`, record `_context_dim`, `_action_input_dim`,
   `_original_img_size`, `_view_names`.
2. Fill in `bind_policy_cameras`. Match the semantics of
   `LPBV2GProvider.bind_policy_cameras` (g_provider.py:153-189) — raise a
   clear error if a required encoder view is not present.
3. Fill in `encode(image_obs_raw, proprio_raw)`:
   - Resize per-view to `_original_img_size` with bilinear interpolation.
   - Build `actions := tile_proprio_as_action(proprio_raw, action_input_dim)`
     (reuse `q_learning/data_util.py::tile_proprio_as_action` once it's
     implemented; until then, inline the tile).
   - Call `self._encoder.encode_batch(images_per_view, proprio, actions)`
     inside `torch.no_grad()`.
   - Return `(B, D_ctx)`.
4. Refactor `dipole/g_provider.py` so `LPBV2GProvider` accepts an
   externally-supplied `SharedFrozenEncoder` instead of building one. Keep
   backward compatibility (if no encoder is passed, build one internally —
   eases the legacy `bce_frozen` config path).
5. **Do not touch** `LPBV2Encoder` or anything under
   `robosuite/discriminator/lpb_v2/` — that package is shared upstream.

## Contract (must match)

- `encode(image_obs_raw, proprio_raw) -> (B, D_ctx)` — see signature in
  the skeleton; do not change.
- `context_dim` property must be available right after `__init__`
  (no lazy build).
- `bind_policy_cameras` mutates internal state; subsequent `encode` calls
  use the bound camera order. Calling `bind_policy_cameras` twice with
  different camera tuples is a hard error.
- All forward calls must be inside `torch.no_grad()`; no parameter has
  `requires_grad`. Assert this in `__init__`.

## Don't do

- Don't add a "fine-tune encoder" mode. The encoder is permanently frozen.
- Don't introduce a separate per-view normalizer; the upstream encoder
  already normalizes internally.
- Don't move the file or rename classes.

## Done criteria

- `python -m py_compile robosuite/pipeline/algorithms/discriminator/encoder.py` passes.
- Existing `train_dipole.py` (legacy DIPOLE) still runs end-to-end with
  `g_mode=bce_frozen` — the refactored `LPBV2GProvider` must remain a drop-in.
- New unit test (you write it under
  `robosuite/discriminator/lpb_v2/tests/test_shared_encoder.py` — see existing
  tests in that folder for style) verifies `encode(...)` shape and that
  `requires_grad` is False on every parameter.

## Dependency on other prompts

None. **Run me first.**
