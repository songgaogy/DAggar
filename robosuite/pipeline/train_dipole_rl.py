"""Hydra entry point for DIPOLE-RL training (DIPOLE + IQL + online disc).

Skeleton only — see docs/prompts/05_integrated_trainer.md for the full
implementation contract.

Top-level flow:
    1. Resolve devices (cuda:0 inference, cuda:1 learner) via
       `train_utils.resolve_algorithm_devices`.
    2. Build SharedFrozenEncoder on cuda:1 from
       `algorithm.discriminator.warm_start_ckpt`.
    3. Build agent / DipoleFlowPolicy as in train_dipole.py.
    4. Build OnlineBCEDiscriminator(encoder=...).
    5. Build IQLLearner(context_dim=encoder.context_dim, action_dim=...).
       If `algorithm.q_learning.warmup_ckpt` is set, load it; else run
       `pretrain_iql_value(warmup_value_steps)` + full updates.
    6. Build AdvantageGProvider(iql, disc, encoder, alpha, beta).
    7. Attach G provider to agent (BCE-frozen during the warmup window if
       `runtime.bootstrap_g_with_frozen_bce`).
    8. Hand all three to DipoleTrainer; start async learner thread.
    9. Run the existing rollout / record_transition loop (reused from
       train_dipole.py); after each `record_transition`, also dispatch the
       transition into the discriminator replay (handled inside trainer).
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":  # pragma: no cover
    main()
