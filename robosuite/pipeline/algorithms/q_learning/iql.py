"""IQL learner with Q-chunking + Q-ensemble expectile V.

Held by `DipoleTrainer` in RL mode. Owns K Q networks, V, target_V and two
optimizers (Q optimizer over all Q params, V optimizer over v.params). The
encoder is INJECTED — this class never instantiates it.

Per-tick step ordering (called by the integrated trainer):
    1. iql.update(step_batch) — Bellman Q + expectile V + polyak target.
    2. Trainer calls iql.compute_advantage_for_batch(actor_batch) which is
       consumed by AdvantageGProvider to construct G for DIPOLE.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import chain
from typing import Any

import torch
from torch import nn

from .common import IQLActorBatch, IQLConfig, IQLStepBatch
from .losses import bellman_q_loss, compute_ensemble_advantage, expectile_v_loss
from .networks import ActionProjector, QHead, TokenProjector, VStateNetwork


class IQLLearner:
    """Q-chunking IQL.

    Args:
        cfg:           IQLConfig.
        state_feature_dim: action-free encoder feature dimension used by V.
        chunk_feature_dim: action-conditioned encoder feature dimension used by Q.
        action_dim: policy action dimension (per step). The Q action projector
            consumes the flattened ``action_horizon * action_dim`` chunk.
        n_tokens: number of patch tokens in the encoder feature (chunk feature
            is ``n_tokens x chunk_token_dim``; state visual block is
            ``n_tokens x state_visual_token_dim``). From ``encoder.num_patches``.
        proprio_dim: width of the proprio block appended to the state feature.
            From ``encoder.proprio_emb_dim``.

    Q now factorizes as: a *shared* Token/Group :class:`TokenProjector` over the
    chunk feature + a *shared* :class:`ActionProjector` over the raw action
    chunk, concatenated and fed to K per-member :class:`QHead`. The projectors
    are the anti-overfit lever and are shared across the ensemble. V is a single
    :class:`VStateNetwork` that owns its own state projector (so ``target_v``
    Polyak-tracks the projector with the head).
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

        if self.n_tokens < 1 or self.chunk_feature_dim % self.n_tokens != 0:
            raise ValueError(
                f"IQLLearner: chunk_feature_dim {self.chunk_feature_dim} not divisible "
                f"by n_tokens {self.n_tokens}."
            )
        if int(cfg.chunk_proj_dim) % self.n_tokens != 0:
            raise ValueError(
                f"IQLLearner: chunk_proj_dim {cfg.chunk_proj_dim} not divisible "
                f"by n_tokens {self.n_tokens}."
            )

        self.chunk_proj_dim = int(cfg.chunk_proj_dim)
        self.action_proj_dim = int(cfg.action_proj_dim)
        self.state_proj_dim = int(cfg.state_proj_dim)
        self.proprio_proj_dim = int(cfg.proprio_proj_dim)
        self.action_flat_dim = int(cfg.action_horizon) * self.action_dim
        activation = str(cfg.proj_activation)

        self.q_ensemble_size = int(cfg.q_ensemble_size)
        self.v_subset_size = int(cfg.v_subset_size)

        # Shared Q-side projectors (one instance for the whole ensemble).
        self.chunk_projector: nn.Module = TokenProjector(
            n_tokens=self.n_tokens,
            token_dim=self.chunk_feature_dim // self.n_tokens,
            out_per_token=self.chunk_proj_dim // self.n_tokens,
            activation=activation,
        ).to(device)
        self.action_projector: nn.Module = ActionProjector(
            action_flat_dim=self.action_flat_dim,
            action_proj_dim=self.action_proj_dim,
            activation=activation,
        ).to(device)
        q_head_in = self.chunk_projector.output_dim + self.action_projector.output_dim
        self.q_ensemble: nn.ModuleList = nn.ModuleList(
            [
                QHead(q_head_in, hidden_dims=tuple(cfg.hidden_dims), activation=activation)
                for _ in range(self.q_ensemble_size)
            ]
        ).to(device)

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

        # The Q optimizer owns the shared projectors + every ensemble head.
        self.q_optim: torch.optim.Optimizer = torch.optim.AdamW(
            chain(
                self.chunk_projector.parameters(),
                self.action_projector.parameters(),
                self.q_ensemble.parameters(),
            ),
            lr=float(cfg.q_lr),
            weight_decay=float(cfg.weight_decay),
        )
        self.v_optim: torch.optim.Optimizer = torch.optim.AdamW(
            self.v.parameters(),
            lr=float(cfg.v_lr),
            weight_decay=float(cfg.weight_decay),
        )

    @property
    def q1(self) -> nn.Module:
        """First critic alias kept for visualization/tests."""
        return self.q_ensemble[0]

    @property
    def q2(self) -> nn.Module:
        """Second critic alias kept for visualization/tests.

        When K=1 this intentionally aliases q1.
        """
        return self.q_ensemble[min(1, self.q_ensemble_size - 1)]

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

    def _q_parameters(self):
        """All Q-side trainable params (shared projectors + ensemble heads)."""
        return chain(
            self.chunk_projector.parameters(),
            self.action_projector.parameters(),
            self.q_ensemble.parameters(),
        )

    def _q_values(
        self, chunk_feature: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Return all critic predictions as (K, B, 1).

        The shared chunk/action projectors run ONCE; only the K heads vary.
        """
        z_chunk = self.chunk_projector(chunk_feature)
        z_action = self.action_projector(action_chunk)
        z = torch.cat([z_chunk, z_action], dim=-1)
        return torch.stack([q(z) for q in self.q_ensemble], dim=0)

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

        q_values = self._q_values(step_batch.q_chunk_feature, step_batch.action_chunk)
        q_loss = bellman_q_loss(q_values, target_q.expand_as(q_values))
        self.q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._q_parameters(),
            float(self.cfg.grad_clip_norm),
        )
        self.q_optim.step()

        with torch.no_grad():
            subset_idx = torch.randperm(
                self.q_ensemble_size,
                device=q_values.device,
            )[: self.v_subset_size]
            q_subset = q_values.detach().index_select(0, subset_idx)
            q_min = q_subset.min(dim=0).values

        v_pred = self.v(step_batch.v_state_feature)
        diff = q_min - v_pred
        v_loss = expectile_v_loss(diff, float(self.cfg.expectile_tau))
        self.v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), float(self.cfg.grad_clip_norm))
        self.v_optim.step()

        self._polyak_update()

        with torch.no_grad():
            q_detached = q_values.detach()
            td_error = (q_detached - target_q.expand_as(q_detached)).abs().mean()

        return {
            "q_loss": float(q_loss.detach().item()),
            "v_loss": float(v_loss.detach().item()),
            "q1_mean": float(q_detached[0].mean().item()),
            "q2_mean": float(q_detached[min(1, self.q_ensemble_size - 1)].mean().item()),
            "q_ensemble_mean": float(q_detached.mean().item()),
            "q_v_subset_min_mean": float(q_min.mean().item()),
            "v_mean": float(v_pred.detach().mean().item()),
            "target_q_mean": float(target_q.detach().mean().item()),
            "td_error_abs_mean": float(td_error.item()),
        }

    def warmup_value_only(self, step_batch: IQLStepBatch) -> dict[str, float]:
        """V-only pretraining step used during offline warmup before Q is
        well-defined. V regresses toward r + γ^H · (1 - done) · target_v(s')
        directly (no min-of-two-Q)."""
        target = self._bootstrap_target(step_batch)
        v_pred = self.v(step_batch.v_state_feature)
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
        q_values = self._q_values(actor_batch.q_chunk_feature, actor_batch.action_chunk)
        v = self.v(actor_batch.v_state_feature)
        return compute_ensemble_advantage(q_values, v)

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        return {
            "chunk_projector": self.chunk_projector.state_dict(),
            "action_projector": self.action_projector.state_dict(),
            "q_ensemble": self.q_ensemble.state_dict(),
            "q_ensemble_size": self.q_ensemble_size,
            "v_subset_size": self.v_subset_size,
            "v": self.v.state_dict(),
            "target_v": self.target_v.state_dict(),
            "q_optim": self.q_optim.state_dict(),
            "v_optim": self.v_optim.state_dict(),
            "cfg": asdict(self.cfg),
            "state_feature_dim": self.state_feature_dim,
            "chunk_feature_dim": self.chunk_feature_dim,
            "action_dim": self.action_dim,
            "n_tokens": self.n_tokens,
            "proprio_dim": self.proprio_dim,
            "chunk_proj_dim": self.chunk_proj_dim,
            "state_proj_dim": self.state_proj_dim,
            "proprio_proj_dim": self.proprio_proj_dim,
            "action_proj_dim": self.action_proj_dim,
        }

    def load_state_dict(self, sd: dict[str, Any], strict: bool = True) -> None:
        if "state_feature_dim" not in sd or "chunk_feature_dim" not in sd:
            raise ValueError(
                "IQLLearner.load_state_dict: checkpoint uses the legacy LPB "
                "single-context critic schema. Re-run offline Q/V warmup with "
                "the nnPU chunk encoder."
            )
        if "chunk_projector" not in sd or "action_projector" not in sd:
            raise ValueError(
                "IQLLearner.load_state_dict: checkpoint predates the Token/Group "
                "dim-reduction projectors (no chunk_projector/action_projector). "
                "Re-run offline Q/V warmup to produce a new checkpoint."
            )
        if strict:
            for key, runtime_val in (
                ("chunk_proj_dim", self.chunk_proj_dim),
                ("state_proj_dim", self.state_proj_dim),
                ("proprio_proj_dim", self.proprio_proj_dim),
                ("action_proj_dim", self.action_proj_dim),
                ("n_tokens", self.n_tokens),
                ("proprio_dim", self.proprio_dim),
            ):
                if int(sd.get(key, -1)) != int(runtime_val):
                    raise ValueError(
                        f"IQLLearner.load_state_dict: {key} mismatch "
                        f"(ckpt={sd.get(key)}, runtime={runtime_val})"
                    )
        if strict:
            if int(sd["state_feature_dim"]) != self.state_feature_dim:
                raise ValueError(
                    "IQLLearner.load_state_dict: state_feature_dim mismatch "
                    f"(ckpt={sd['state_feature_dim']}, runtime={self.state_feature_dim})"
                )
            if int(sd["chunk_feature_dim"]) != self.chunk_feature_dim:
                raise ValueError(
                    "IQLLearner.load_state_dict: chunk_feature_dim mismatch "
                    f"(ckpt={sd['chunk_feature_dim']}, runtime={self.chunk_feature_dim})"
                )
            if int(sd.get("action_dim", -1)) != self.action_dim:
                raise ValueError(
                    f"IQLLearner.load_state_dict: action_dim mismatch "
                    f"(ckpt={sd.get('action_dim')}, runtime={self.action_dim})"
                )
        if "q_ensemble" not in sd:
            raise ValueError(
                "IQLLearner.load_state_dict: checkpoint uses the old two-Q schema "
                "(q1/q2). Re-run offline Q-ensemble warmup to produce a new checkpoint."
            )
        ckpt_k = int(sd.get("q_ensemble_size", -1))
        if ckpt_k != self.q_ensemble_size:
            raise ValueError(
                f"IQLLearner.load_state_dict: q_ensemble_size mismatch "
                f"(ckpt={ckpt_k}, runtime={self.q_ensemble_size})"
            )
        self.chunk_projector.load_state_dict(sd["chunk_projector"])
        self.action_projector.load_state_dict(sd["action_projector"])
        self.q_ensemble.load_state_dict(sd["q_ensemble"])
        self.v.load_state_dict(sd["v"])
        self.target_v.load_state_dict(sd["target_v"])
        self.q_optim.load_state_dict(sd["q_optim"])
        self.v_optim.load_state_dict(sd["v_optim"])
