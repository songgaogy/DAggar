"""V-only IQL learner (no Q head).

Held by `DipoleTrainer` in RL mode. Owns a V ensemble, its target ensemble and
a single optimizer. The encoder is INJECTED — this class never instantiates it.

Under the deterministic dynamics + single-action-per-state coverage here, the
Q step of IQL is redundant: ``Q(s,a) = r + γ^H V(s')`` holds exactly, so Q
cancels out of the value fit and the value can be learned V-only by a TD
backup. The per-step advantage is the TD residual on V. See
``pipeline/V_ONLY_ADVANTAGE_DESIGN.md``.

The value fit is the §4 finalized method (not plain MSE):
    - **Optimism**: each head regresses onto the 1-step bootstrap target by an
      *expectile* loss (``expectile_tau`` > 0.5), turning the behavior value
      ``V^β`` into the optimistic ``V*``.
    - **Soft-LCB pessimism guardrail**: the bootstrap target and the advantage
      read-out both use ``V_lcb = mean_k V_k − β·std_k V_k`` over the N-head
      ensemble (``ensemble_lcb_beta``). Heads are diversified by a per-head
      Bernoulli bootstrap mask so the std does not collapse.
The TD target stays **1-step** (n-step propagation was tried and reverted); λ
survives only in the read-out GAE (vis_qv / offline advantage). ``N == 1``
recovers the legacy single-head 1-step backup (std ≡ 0; expectile still active).

Per-tick step ordering (called by the integrated trainer):
    1. iql.update(step_batch) — each V head regresses onto the shared LCB
       bootstrap target (expectile-TD) + polyak target update.
    2. Trainer/AdvantageGProvider calls iql.compute_td_advantage(...) to
       produce the TD-residual advantage (on ``V_lcb``) consumed by DIPOLE.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .common import IQLConfig, IQLStepBatch
from .losses import expectile_v_loss
from .networks import VEnsemble


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

    V is a :class:`VEnsemble` of ``cfg.v_ensemble_size`` independent heads, each
    owning its own state projector (so ``target_v`` Polyak-tracks the projectors
    with the heads). The scalar value consumed everywhere is the soft-LCB
    ``V_lcb = mean_k − β·std_k`` (:meth:`v_lcb` / :meth:`target_v_lcb`).
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
        self.ensemble_size = int(cfg.v_ensemble_size)
        self.lcb_beta = float(cfg.ensemble_lcb_beta)
        activation = str(cfg.proj_activation)

        def _build_ensemble() -> VEnsemble:
            return VEnsemble(
                ensemble_size=self.ensemble_size,
                state_feature_dim=self.state_feature_dim,
                n_tokens=self.n_tokens,
                proprio_dim=self.proprio_dim,
                state_proj_dim=self.state_proj_dim,
                proprio_proj_dim=self.proprio_proj_dim,
                hidden_dims=tuple(cfg.hidden_dims),
                activation=activation,
            ).to(device)

        self.v: nn.Module = _build_ensemble()
        self.target_v: nn.Module = _build_ensemble()
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

    def _lcb(self, per_head: torch.Tensor) -> torch.Tensor:
        """Soft-LCB reduction ``mean_k − β·std_k`` over the head axis.

        ``per_head`` is (B, N); returns (B, 1). Population std (unbiased=False)
        so N==1 gives std==0 (β inert), matching the single-head V exactly.
        """
        mean = per_head.mean(dim=-1, keepdim=True)
        std = per_head.std(dim=-1, unbiased=False, keepdim=True)
        return mean - self.lcb_beta * std

    def v_lcb(self, state_feature: torch.Tensor) -> torch.Tensor:
        """Online-ensemble soft-LCB value ``V_lcb(s)`` -> (B, 1)."""
        return self._lcb(self.v(state_feature))

    def target_v_lcb(self, state_feature: torch.Tensor) -> torch.Tensor:
        """Target-ensemble soft-LCB value -> (B, 1)."""
        return self._lcb(self.target_v(state_feature))

    def _bootstrap_target(self, step_batch: IQLStepBatch) -> torch.Tensor:
        """Bellman target r + γ^H · (1 - done) · V_lcb_target(s')  -> (B, 1).

        Shared across heads (each head regresses onto the same LCB target).
        """
        bootstrap_discount = float(self.cfg.discount) ** int(self.cfg.action_horizon)
        with torch.no_grad():
            v_next = self.target_v_lcb(step_batch.next_v_state_feature)
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
        """One V-only expectile-TD optimization step + polyak target update.

        Every head regresses onto the shared 1-step LCB bootstrap target
        ``r + γ^H · (1 - done) · V_lcb_target(s')`` by an *expectile* loss
        (τ=``expectile_tau``), each head weighted by an independent per-sample
        Bernoulli bootstrap mask (keep-prob ``ensemble_bootstrap_prob``) so the
        heads diverge and the LCB std stays non-degenerate. Returns metrics:
        v_loss, v_mean (V_lcb), target_mean, td_error_abs_mean, v_std_mean
        (mean per-sample head std — the LCB diversity monitor).
        """
        target = self._bootstrap_target(step_batch)             # (B, 1) shared
        v_all = self.v(step_batch.v_state_feature)              # (B, N)
        diff = target - v_all                                   # (B, N) broadcast
        mask = (
            torch.rand_like(v_all) < float(self.cfg.ensemble_bootstrap_prob)
        ).to(v_all.dtype)
        # Guard the degenerate all-zero mask (rare, tiny batch): fall back to
        # unmasked so the step is never a no-op / NaN.
        if float(mask.sum().item()) == 0.0:
            mask = torch.ones_like(v_all)
        v_loss = expectile_v_loss(diff, float(self.cfg.expectile_tau), weights=mask)
        self.v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v.parameters(), float(self.cfg.grad_clip_norm))
        self.v_optim.step()
        self._polyak_update()

        with torch.no_grad():
            v_lcb = self._lcb(v_all.detach())
            td_error = (v_lcb - target.detach()).abs().mean()
            v_std = v_all.detach().std(dim=-1, unbiased=False).mean()

        return {
            "v_loss": float(v_loss.detach().item()),
            "v_mean": float(v_lcb.mean().item()),
            "target_mean": float(target.detach().mean().item()),
            "td_error_abs_mean": float(td_error.item()),
            "v_std_mean": float(v_std.item()),
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
        """TD-residual advantage A = r + γ^H·(1-done)·V_lcb_target(s') - V_lcb(s).

        Shapes: features (B, D_state); rewards/dones (B, 1) or (B,). Returns
        (B,). This is the same backup V is trained on (:meth:`update`), read out
        on the soft-LCB value, so the residual measures how much
        better-than-baseline the transition is.
        """
        bootstrap_discount = float(self.cfg.discount) ** int(self.cfg.action_horizon)
        v_s = self.v_lcb(v_state_feature)
        v_sp = self.target_v_lcb(next_v_state_feature)
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
            "v_ensemble_size": self.ensemble_size,
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
                ("v_ensemble_size", self.ensemble_size),
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
