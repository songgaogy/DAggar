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

from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .common import IQLActorBatch, IQLConfig, IQLStepBatch
from .losses import bellman_q_loss, compute_advantage, expectile_v_loss
from .networks import QChunkNetwork, VNetwork


class IQLLearner:
    """Q-chunking IQL.

    Args:
        cfg:           IQLConfig.
        context_dim:   D_ctx of the frozen encoder.
        action_dim:    per-step POLICY action dim D_a (NOT encoder
                       action_dim_per_step). Q input = D_ctx + H * D_a.
    """

    def __init__(self, cfg: IQLConfig, context_dim: int, action_dim: int) -> None:
        self.cfg = cfg
        self.context_dim = int(context_dim)
        self.action_dim = int(action_dim)
        device = cfg.device

        self.q1: nn.Module = QChunkNetwork(
            context_dim=self.context_dim,
            action_dim=self.action_dim,
            action_horizon=int(cfg.action_horizon),
            hidden_dims=tuple(cfg.hidden_dims),
        ).to(device)
        self.q2: nn.Module = QChunkNetwork(
            context_dim=self.context_dim,
            action_dim=self.action_dim,
            action_horizon=int(cfg.action_horizon),
            hidden_dims=tuple(cfg.hidden_dims),
        ).to(device)
        self.v: nn.Module = VNetwork(
            context_dim=self.context_dim,
            hidden_dims=tuple(cfg.hidden_dims),
        ).to(device)
        self.target_v: nn.Module = VNetwork(
            context_dim=self.context_dim,
            hidden_dims=tuple(cfg.hidden_dims),
        ).to(device)
        self.target_v.load_state_dict(self.v.state_dict())
        for p in self.target_v.parameters():
            p.requires_grad_(False)

        self.q_optim: torch.optim.Optimizer = torch.optim.AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=float(cfg.q_lr),
            weight_decay=float(cfg.weight_decay),
        )
        self.v_optim: torch.optim.Optimizer = torch.optim.AdamW(
            self.v.parameters(),
            lr=float(cfg.v_lr),
            weight_decay=float(cfg.weight_decay),
        )

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _bootstrap_target(self, step_batch: IQLStepBatch) -> torch.Tensor:
        """Bellman target r + γ^H · (1 - done) · target_v(s')."""
        bootstrap_discount = float(self.cfg.discount) ** int(self.cfg.action_horizon)
        with torch.no_grad():
            v_next = self.target_v(step_batch.next_context)
            target = step_batch.rewards + bootstrap_discount * (1.0 - step_batch.dones) * v_next
        return target

    @torch.no_grad()
    def _polyak_update(self) -> None:
        tau_p = float(self.cfg.target_polyak)
        for tgt, src in zip(self.target_v.parameters(), self.v.parameters()):
            tgt.data.mul_(1.0 - tau_p).add_(src.data, alpha=tau_p)

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def update(self, step_batch: IQLStepBatch) -> dict[str, float]:
        """One Q + V optimization step + polyak target update.

        Returns metrics dict: q_loss, v_loss, q1_mean, q2_mean, v_mean,
        target_q_mean, td_error_abs_mean.
        """
        target_q = self._bootstrap_target(step_batch)

        q1_pred = self.q1(step_batch.context, step_batch.action_chunk)
        q2_pred = self.q2(step_batch.context, step_batch.action_chunk)
        q_loss = bellman_q_loss(q1_pred, target_q) + bellman_q_loss(q2_pred, target_q)
        self.q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            float(self.cfg.grad_clip_norm),
        )
        self.q_optim.step()

        with torch.no_grad():
            q_min = torch.min(q1_pred.detach(), q2_pred.detach())

        v_pred = self.v(step_batch.context)
        diff = q_min - v_pred
        v_loss = expectile_v_loss(diff, float(self.cfg.expectile_tau))
        self.v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), float(self.cfg.grad_clip_norm))
        self.v_optim.step()

        self._polyak_update()

        with torch.no_grad():
            td_error = (q1_pred.detach() - target_q).abs().mean()

        return {
            "q_loss": float(q_loss.detach().item()),
            "v_loss": float(v_loss.detach().item()),
            "q1_mean": float(q1_pred.detach().mean().item()),
            "q2_mean": float(q2_pred.detach().mean().item()),
            "v_mean": float(v_pred.detach().mean().item()),
            "target_q_mean": float(target_q.detach().mean().item()),
            "td_error_abs_mean": float(td_error.item()),
        }

    def warmup_value_only(self, step_batch: IQLStepBatch) -> dict[str, float]:
        """V-only pretraining step used during offline warmup before Q is
        well-defined. V regresses toward r + γ^H · (1 - done) · target_v(s')
        directly (no min-of-two-Q)."""
        target = self._bootstrap_target(step_batch)
        v_pred = self.v(step_batch.context)
        v_loss = torch.nn.functional.mse_loss(v_pred, target)
        self.v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), float(self.cfg.grad_clip_norm))
        self.v_optim.step()
        self._polyak_update()
        return {
            "v_loss": float(v_loss.detach().item()),
            "v_mean": float(v_pred.detach().mean().item()),
            "target_mean": float(target.detach().mean().item()),
        }

    # ------------------------------------------------------------------ #
    # Advantage scoring (consumed by AdvantageGProvider)                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_advantage_for_batch(self, actor_batch: IQLActorBatch) -> torch.Tensor:
        """Return A(s, a_chunk) shape (B,). Pure inference, no grads."""
        q1 = self.q1(actor_batch.context, actor_batch.action_chunk_raw)
        q2 = self.q2(actor_batch.context, actor_batch.action_chunk_raw)
        v = self.v(actor_batch.context)
        return compute_advantage(q1, q2, v)

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        return {
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "v": self.v.state_dict(),
            "target_v": self.target_v.state_dict(),
            "q_optim": self.q_optim.state_dict(),
            "v_optim": self.v_optim.state_dict(),
            "cfg": asdict(self.cfg),
            "context_dim": self.context_dim,
            "action_dim": self.action_dim,
        }

    def load_state_dict(self, sd: dict[str, Any], strict: bool = True) -> None:
        if strict:
            if int(sd.get("context_dim", -1)) != self.context_dim:
                raise ValueError(
                    f"IQLLearner.load_state_dict: context_dim mismatch "
                    f"(ckpt={sd.get('context_dim')}, runtime={self.context_dim})"
                )
            if int(sd.get("action_dim", -1)) != self.action_dim:
                raise ValueError(
                    f"IQLLearner.load_state_dict: action_dim mismatch "
                    f"(ckpt={sd.get('action_dim')}, runtime={self.action_dim})"
                )
        self.q1.load_state_dict(sd["q1"])
        self.q2.load_state_dict(sd["q2"])
        self.v.load_state_dict(sd["v"])
        self.target_v.load_state_dict(sd["target_v"])
        self.q_optim.load_state_dict(sd["q_optim"])
        self.v_optim.load_state_dict(sd["v_optim"])
