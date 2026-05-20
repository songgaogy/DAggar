"""IQL learner with Q-chunking + expectile V.

Held by `DipoleTrainer` in RL mode. Owns Q1, Q2, V, target_V and two
optimizers (Q optimizer over q1.params + q2.params, V optimizer over
v.params). The encoder is INJECTED — this class never instantiates it.

Per-tick step ordering (called by the integrated trainer):
    1. iql.update(step_batch) — Bellman Q + expectile V + polyak target.
    2. Trainer calls iql.compute_advantage_for_batch(actor_batch) which is
       consumed by AdvantageGProvider to construct G for DIPOLE.
"""

from __future__ import annotations

import torch
from torch import nn

from .common import IQLActorBatch, IQLConfig, IQLStepBatch


class IQLLearner:
    """Q-chunking IQL.

    Args:
        cfg:           IQLConfig.
        context_dim:   D_ctx of the frozen encoder.
        action_dim:    per-step action dim D_a.
    """

    def __init__(self, cfg: IQLConfig, context_dim: int, action_dim: int) -> None:
        self.cfg = cfg
        self.context_dim = context_dim
        self.action_dim = action_dim

        # Networks (instantiate in implementation):
        #   self.q1 = QChunkNetwork(context_dim, action_dim, cfg.action_horizon,
        #                           cfg.hidden_dims).to(cfg.device)
        #   self.q2 = QChunkNetwork(...)
        #   self.v        = VNetwork(context_dim, cfg.hidden_dims).to(cfg.device)
        #   self.target_v = VNetwork(...).to(cfg.device); copy v -> target_v.
        #
        # Optimizers (AdamW, weight_decay=cfg.weight_decay):
        #   self.q_optim = AdamW(q1.params + q2.params, lr=cfg.q_lr, ...)
        #   self.v_optim = AdamW(v.params,             lr=cfg.v_lr, ...)
        self.q1: nn.Module | None = None
        self.q2: nn.Module | None = None
        self.v: nn.Module | None = None
        self.target_v: nn.Module | None = None
        self.q_optim: torch.optim.Optimizer | None = None
        self.v_optim: torch.optim.Optimizer | None = None

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def update(self, step_batch: IQLStepBatch) -> dict[str, float]:
        """One Q + V optimization step + polyak target update.

        Returns metrics dict: q_loss, v_loss, q1_mean, q2_mean, v_mean,
        target_q_mean, td_error_abs_mean.
        """
        raise NotImplementedError

    def warmup_value_only(self, step_batch: IQLStepBatch) -> dict[str, float]:
        """V-only pretraining step used during offline warmup before Q is
        well-defined (see warmup.py)."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Advantage scoring (consumed by AdvantageGProvider)                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_advantage_for_batch(self, actor_batch: IQLActorBatch) -> torch.Tensor:
        """Return A(s, a_chunk) shape (B,). Pure inference, no grads."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, sd: dict, strict: bool = True) -> None:
        raise NotImplementedError
