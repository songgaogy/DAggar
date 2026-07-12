"""V-only IQL learner (no Q head).

Held by `DipoleTrainer` in RL mode. Owns V, target_V and a single V optimizer.
The encoder is INJECTED — this class never instantiates it.

Under the deterministic dynamics + single-action-per-state coverage here, the
Q step of IQL is redundant: ``Q(s,a) = r + γ^H V(s')`` holds exactly, so Q
cancels out of the value fit and the value can be learned V-only by a TD
backup. The per-step advantage is the TD residual on V. See
``pipeline/V_ONLY_ADVANTAGE_DESIGN.md``.

Per-tick step ordering (called by the integrated trainer):
    1. iql.update(step_batch) — V regresses onto the bootstrap target
       (MSE-TD) + polyak target update.
    2. Trainer/AdvantageGProvider calls iql.compute_td_advantage(...) to
       produce the TD-residual advantage consumed by DIPOLE.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .common import IQLConfig, IQLStepBatch
from .networks import VStateNetwork


class IQLLearner:
    """V-only value learner for DIPOLE.

    Args:
        cfg:           IQLConfig.
        state_feature_dim: action-free encoder feature dimension used by V.
        chunk_feature_dim: action-conditioned encoder feature dimension. Kept
            only as checkpoint metadata (identifies the encoder that produced
            the features); the value function does not consume it.
        action_dim: policy action dimension (per step). Kept as metadata.
        n_tokens: number of patch tokens in the encoder feature (state visual
            block is ``n_tokens x state_visual_token_dim``). From
            ``encoder.num_patches``.
        proprio_dim: width of the proprio block appended to the state feature.
            From ``encoder.proprio_emb_dim``.

    V is a single :class:`VStateNetwork` that owns its own state projector (so
    ``target_v`` Polyak-tracks the projector with the head).
    """

    def __init__(
        self,
        cfg: IQLConfig,
        state_feature_dim: int,
        chunk_feature_dim: int,
        action_dim: int,
        n_tokens: int,
        proprio_dim: int,
    ) -> None:
        self.cfg = cfg
        self.state_feature_dim = int(state_feature_dim)
        self.chunk_feature_dim = int(chunk_feature_dim)
        self.action_dim = int(action_dim)
        self.n_tokens = int(n_tokens)
        self.proprio_dim = int(proprio_dim)
        device = cfg.device

        self.state_proj_dim = int(cfg.state_proj_dim)
        self.proprio_proj_dim = int(cfg.proprio_proj_dim)
        activation = str(cfg.proj_activation)

        self.v: nn.Module = VStateNetwork(
            state_feature_dim=self.state_feature_dim,
            n_tokens=self.n_tokens,
            proprio_dim=self.proprio_dim,
            state_proj_dim=self.state_proj_dim,
            proprio_proj_dim=self.proprio_proj_dim,
            hidden_dims=tuple(cfg.hidden_dims),
            activation=activation,
        ).to(device)
        self.target_v: nn.Module = VStateNetwork(
            state_feature_dim=self.state_feature_dim,
            n_tokens=self.n_tokens,
            proprio_dim=self.proprio_dim,
            state_proj_dim=self.state_proj_dim,
            proprio_proj_dim=self.proprio_proj_dim,
            hidden_dims=tuple(cfg.hidden_dims),
            activation=activation,
        ).to(device)
        self.target_v.load_state_dict(self.v.state_dict())
        for p in self.target_v.parameters():
            p.requires_grad_(False)

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
            v_next = self.target_v(step_batch.next_v_state_feature)
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
        """One V-only MSE-TD optimization step + polyak target update.

        V regresses directly toward the bootstrap target
        ``r + γ^H · (1 - done) · target_v(s')`` by plain MSE (no Q). Returns
        metrics dict: v_loss, v_mean, target_mean, td_error_abs_mean.
        """
        target = self._bootstrap_target(step_batch)
        v_pred = self.v(step_batch.v_state_feature)
        v_loss = torch.nn.functional.mse_loss(v_pred, target)
        self.v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), float(self.cfg.grad_clip_norm))
        self.v_optim.step()
        self._polyak_update()

        with torch.no_grad():
            td_error = (v_pred.detach() - target.detach()).abs().mean()

        return {
            "v_loss": float(v_loss.detach().item()),
            "v_mean": float(v_pred.detach().mean().item()),
            "target_mean": float(target.detach().mean().item()),
            "td_error_abs_mean": float(td_error.item()),
        }

    # ------------------------------------------------------------------ #
    # Advantage scoring (consumed by AdvantageGProvider)                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_td_advantage(
        self,
        v_state_feature: torch.Tensor,
        next_v_state_feature: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """TD-residual advantage A = r + γ^H·(1-done)·target_v(s') - v(s).

        Shapes: features (B, D_state); rewards/dones (B, 1) or (B,). Returns
        (B,). This is the same backup V is trained on (:meth:`update`), so the
        residual measures how much better-than-baseline the transition is.
        """
        bootstrap_discount = float(self.cfg.discount) ** int(self.cfg.action_horizon)
        v_s = self.v(v_state_feature)
        v_sp = self.target_v(next_v_state_feature)
        rewards = rewards.reshape_as(v_s)
        dones = dones.reshape_as(v_s)
        advantage = rewards + bootstrap_discount * (1.0 - dones) * v_sp - v_s
        return advantage.reshape(-1)

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        return {
            "v": self.v.state_dict(),
            "target_v": self.target_v.state_dict(),
            "v_optim": self.v_optim.state_dict(),
            "cfg": asdict(self.cfg),
            "state_feature_dim": self.state_feature_dim,
            "chunk_feature_dim": self.chunk_feature_dim,
            "action_dim": self.action_dim,
            "n_tokens": self.n_tokens,
            "proprio_dim": self.proprio_dim,
            "state_proj_dim": self.state_proj_dim,
            "proprio_proj_dim": self.proprio_proj_dim,
        }

    def load_state_dict(self, sd: dict[str, Any], strict: bool = True) -> None:
        if "q_ensemble" in sd or "q1" in sd:
            raise ValueError(
                "IQLLearner.load_state_dict: checkpoint contains a Q head "
                "(q_ensemble/q1). This is the V-only learner — re-run the "
                "V-only offline warmup to produce a new checkpoint."
            )
        if "state_feature_dim" not in sd or "v" not in sd:
            raise ValueError(
                "IQLLearner.load_state_dict: checkpoint uses a legacy schema "
                "with no V-only state. Re-run offline V warmup to produce a "
                "new checkpoint."
            )
        if strict:
            for key, runtime_val in (
                ("state_proj_dim", self.state_proj_dim),
                ("proprio_proj_dim", self.proprio_proj_dim),
                ("n_tokens", self.n_tokens),
                ("proprio_dim", self.proprio_dim),
            ):
                if int(sd.get(key, -1)) != int(runtime_val):
                    raise ValueError(
                        f"IQLLearner.load_state_dict: {key} mismatch "
                        f"(ckpt={sd.get(key)}, runtime={runtime_val})"
                    )
            if int(sd["state_feature_dim"]) != self.state_feature_dim:
                raise ValueError(
                    "IQLLearner.load_state_dict: state_feature_dim mismatch "
                    f"(ckpt={sd['state_feature_dim']}, runtime={self.state_feature_dim})"
                )
        self.v.load_state_dict(sd["v"])
        self.target_v.load_state_dict(sd["target_v"])
        self.v_optim.load_state_dict(sd["v_optim"])
