"""Offline Q/V warmup entry point for IQL.

Mirrors baseline awr/models/build_awr_qv_cache.py. Loads:
  - the frozen flow-policy actor (from runtime.init_checkpoint),
  - offline expert demos (HDF5),
  - success + failure rollout caches (optional),
  - the SharedFrozenEncoder + (optionally) a warm-started OnlineBCE disc,
then runs `iql.warmup_value_only(...)` for N steps and `iql.update(...)`
for M steps, finally dumping `iql_state.pt` to disk for the online phase
to pick up via `algorithm.q_learning.warmup_ckpt`.

Run as a module:
    python -m robosuite.pipeline.algorithms.q_learning.warmup \
        --config robosuite/pipeline/config/train_dipole_rl.yaml \
        --steps 20000 --device cuda:1
"""

from __future__ import annotations


def main() -> None:
    """CLI entry. Implementation should:

    1. Parse Hydra/argparse args (config path, output path, step counts).
    2. Build SharedFrozenEncoder from algorithm.discriminator.warm_start_ckpt.
    3. Build OnlineBCEDiscriminator (warm-started; frozen during warmup so
       the intrinsic reward is at least consistent).
    4. Build IQLLearner with context_dim from encoder and action_dim from cfg.
    5. Load offline demos + (optional) success/fail caches into the base
       transition store; build IQLReplayBuffer on top.
    6. Loop warmup_value_steps: iql.warmup_value_only(step_batch).
    7. Loop warmup_full_steps: iql.update(step_batch).
    8. Dump iql.state_dict() + cfg + encoder_meta to {output}/iql_state.pt.
    """
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    main()
