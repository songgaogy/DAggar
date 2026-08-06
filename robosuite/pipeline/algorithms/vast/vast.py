"""VAST value-stitching learner used by DIPOLE.

The learner jointly trains the goal-conditioned macro-return model
``G(s, s_k, k)`` and an expectile value model. The shared dynamics encoder is
injected and remains frozen. ``single_vast`` uses the unchanged single-head
path, while ``indep_ensemble`` trains independent V / target-V pairs and reads
their mean value.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any
import warnings

import torch
from torch import nn

from .common import VASTConfig, VASTStepBatch
from .losses import expectile_v_loss
from .networks import GoalConditionedValueNetwork, VEnsemble


class VASTLearner:
    """VAST goal-conditioned return and value learner for DIPOLE.

    Args:
        cfg:           VASTConfig.
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
    owning its own projector, target head, optimizer, and gradient clipping.
    The scalar value consumed everywhere is the head mean; ensemble spread is
    retained only as a diagnostic.
    """

    def __init__(
        self,
        cfg: VASTConfig,
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
        if not str(device).startswith("cuda"):
            raise ValueError(
                "VASTLearner requires a CUDA device; CPU tensor execution is disabled "
                f"for this project (got device={device!r})."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "VASTLearner requires CUDA, but torch.cuda.is_available() is False."
            )

        self.state_proj_dim = int(cfg.state_proj_dim)
        self.proprio_proj_dim = int(cfg.proprio_proj_dim)
        self.vast_v_mode = str(cfg.vast_v_mode)
        self.ensemble_size = (
            1 if self.vast_v_mode == "single_vast" else int(cfg.v_ensemble_size)
        )
        activation = str(cfg.proj_activation)

        def _build_ensemble(ensemble_size: int) -> VEnsemble:
            return VEnsemble(
                ensemble_size=ensemble_size,
                state_feature_dim=self.state_feature_dim,
                n_tokens=self.n_tokens,
                proprio_dim=self.proprio_dim,
                state_proj_dim=self.state_proj_dim,
                proprio_proj_dim=self.proprio_proj_dim,
                hidden_dims=tuple(cfg.hidden_dims),
                activation=activation,
            ).to(device)

        if self.vast_v_mode == "indep_ensemble":
            # Preserve the single_vast G initialization for matched-seed
            # ablations. V modules initialize on CPU before they move to CUDA,
            # so only the CPU RNG stream must be advanced as a single-head pair.
            initial_rng_state = torch.random.get_rng_state()
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(initial_rng_state)
                _build_ensemble(1)
                _build_ensemble(1)
                single_v_g_rng_state = torch.random.get_rng_state()
        else:
            single_v_g_rng_state = None

        self.v: VEnsemble = _build_ensemble(self.ensemble_size)
        self.target_v: VEnsemble = _build_ensemble(self.ensemble_size)
        self.target_v.load_state_dict(self.v.state_dict())
        for p in self.target_v.parameters():
            p.requires_grad_(False)

        self.v_optims: list[torch.optim.Optimizer] = []
        if self.vast_v_mode == "single_vast":
            self.v_optim: torch.optim.Optimizer | None = torch.optim.AdamW(
                self.v.parameters(),
                lr=float(cfg.v_lr),
                weight_decay=float(cfg.weight_decay),
            )
        else:
            self.v_optim = None
            self.v_optims = [
                torch.optim.AdamW(
                    head.parameters(),
                    lr=float(cfg.v_lr),
                    weight_decay=float(cfg.weight_decay),
                )
                for head in self.v.heads
            ]

        if single_v_g_rng_state is not None:
            torch.random.set_rng_state(single_v_g_rng_state)

        self.g: nn.Module = GoalConditionedValueNetwork(
            state_feature_dim=self.state_feature_dim,
            n_tokens=self.n_tokens,
            proprio_dim=self.proprio_dim,
            state_proj_dim=self.state_proj_dim,
            proprio_proj_dim=self.proprio_proj_dim,
            hidden_dims=tuple(cfg.hidden_dims),
            activation=activation,
        ).to(device)
        self.g_optim: torch.optim.Optimizer = torch.optim.AdamW(
            self.g.parameters(),
            lr=float(cfg.g_lr),
            weight_decay=float(cfg.weight_decay),
        )

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _head_mean(per_head: torch.Tensor) -> torch.Tensor:
        """Reduce per-head values ``(B, N)`` to their mean ``(B, 1)``."""
        return per_head.mean(dim=-1, keepdim=True)

    def v_value(self, state_feature: torch.Tensor) -> torch.Tensor:
        """Online scalar V: single head or independent-ensemble mean."""
        return self._head_mean(self.v(state_feature))

    def target_v_value(self, state_feature: torch.Tensor) -> torch.Tensor:
        """Target scalar V: single head or independent-ensemble mean."""
        return self._head_mean(self.target_v(state_feature))

    def g_value(
        self,
        state_feature: torch.Tensor,
        future_state_feature: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the VAST macro-return model ``G(s, s_k, k)`` -> (B, 1)."""
        return self.g(state_feature, future_state_feature, k)

    def _bootstrap_target(self, step_batch: VASTStepBatch) -> torch.Tensor:
        """Bellman targets for single V ``(B,1)`` or independent heads ``(B,N)``."""
        bootstrap_discount = float(self.cfg.discount) ** int(self.cfg.action_horizon)
        with torch.no_grad():
            v_next = self.target_v(step_batch.next_v_state_feature)
            target = step_batch.rewards + bootstrap_discount * (1.0 - step_batch.dones) * v_next
        return target

    @torch.no_grad()
    def _polyak_update(self) -> None:
        tau_p = float(self.cfg.target_polyak)
        if self.vast_v_mode == "single_vast":
            for tgt, src in zip(self.target_v.parameters(), self.v.parameters()):
                tgt.data.mul_(1.0 - tau_p).add_(src.data, alpha=tau_p)
            return
        for target_head, online_head in zip(self.target_v.heads, self.v.heads):
            for tgt, src in zip(target_head.parameters(), online_head.parameters()):
                tgt.data.mul_(1.0 - tau_p).add_(src.data, alpha=tau_p)

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def update(self, step_batch: VASTStepBatch) -> dict[str, float]:
        """Run one joint VAST G/V update and Polyak-update target V.

        G uses Monte-Carlo and composition losses. V uses expectile regression
        against detached stitched targets. Independent ensemble heads use the
        full batch and their paired target heads.
        """
        return self._update_vast(step_batch)

    @staticmethod
    def _require_vast_batch(step_batch: VASTStepBatch) -> None:
        missing = [
            name
            for name in (
                "future_v_state_feature",
                "intermediate_v_state_feature",
                "k",
                "j",
                "k_step_returns",
                "mc_mask",
                "future_dones",
            )
            if getattr(step_batch, name) is None
        ]
        if missing:
            raise ValueError(
                "VAST update requires replay-provided fields: " + ", ".join(missing)
            )

    def _vast_stitched_target(self, step_batch: VASTStepBatch) -> torch.Tensor:
        """Detached stitched targets: ``(B,1)`` single or ``(B,N)`` ensemble."""
        self._require_vast_batch(step_batch)
        assert step_batch.future_v_state_feature is not None
        assert step_batch.k is not None
        assert step_batch.future_dones is not None
        with torch.no_grad():
            g_tk = self.g_value(
                step_batch.v_state_feature,
                step_batch.future_v_state_feature,
                step_batch.k,
            )
            v_future = self.target_v(step_batch.future_v_state_feature)
            discount = torch.pow(
                torch.full_like(g_tk, float(self.cfg.discount)),
                step_batch.k.reshape_as(g_tk).to(g_tk.dtype)
                * int(self.cfg.action_horizon),
            )
            return g_tk + discount * (1.0 - step_batch.future_dones.reshape_as(g_tk)) * v_future

    def _update_vast(self, step_batch: VASTStepBatch) -> dict[str, float]:
        """Joint VAST G + expectile-V update on one macro-horizon batch.

        G is trained by Monte-Carlo endpoint and compositional consistency
        losses. Every k >= 1 path has a direct MC target; composition is defined
        only for k >= 2. The V target is detached, so V gradients never update G.
        """
        self._require_vast_batch(step_batch)

        k = step_batch.k.reshape(-1, 1)
        j = step_batch.j.reshape(-1, 1)
        mc_mask = step_batch.mc_mask.reshape(-1, 1).to(
            step_batch.v_state_feature.dtype
        )
        comp_mask = (k >= 2).to(step_batch.v_state_feature.dtype)
        returns = step_batch.k_step_returns.reshape(-1, 1)

        # G-composition loss. Evaluate j and k-j only for legal k >= 2 rows so
        # a k=1 sample never constructs the undefined G(..., 0) term.
        g_tk = self.g_value(step_batch.v_state_feature, step_batch.future_v_state_feature, k)
        comp_rows = comp_mask.reshape(-1).to(torch.bool)
        if bool(comp_rows.any().item()):
            g_tj = self.g_value(
                step_batch.v_state_feature[comp_rows],
                step_batch.intermediate_v_state_feature[comp_rows],
                j[comp_rows],
            )
            g_jk = self.g_value(
                step_batch.intermediate_v_state_feature[comp_rows],
                step_batch.future_v_state_feature[comp_rows],
                k[comp_rows] - j[comp_rows],
            )
            j_discount = torch.pow(
                torch.full_like(g_tj, float(self.cfg.discount)),
                j[comp_rows].to(g_tj.dtype) * int(self.cfg.action_horizon),
            )
            composition_rhs = g_tj + j_discount * g_jk
            comp_error = g_tk[comp_rows] - composition_rhs
            # Preserve the established batch-diluted composition scale.
            g_comp_loss = comp_error.square().sum() / g_tk.numel()
        else:
            comp_error = g_tk.new_zeros((0, 1))
            g_comp_loss = g_tk.sum() * 0.0

        # G-MC loss
        mc_sq = (g_tk - returns).square()
        g_mc_loss = (mc_sq * mc_mask).mean()

        g_loss = g_mc_loss + float(self.cfg.vast_comp_coef) * g_comp_loss

        # Build the detached target before stepping G so G and V are updated
        # from one coherent parameter snapshot.
        stitched_target = self._vast_stitched_target(step_batch)

        self.g_optim.zero_grad(set_to_none=True)
        if bool(mc_mask.bool().any().item()) or bool(comp_mask.bool().any().item()):
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.g.parameters(), float(self.cfg.grad_clip_norm))
            self.g_optim.step()

        v_head_losses: list[torch.Tensor] = []
        v_predictions: list[torch.Tensor] = []
        if self.vast_v_mode == "single_vast":
            assert self.v_optim is not None
            v_all = self.v(step_batch.v_state_feature)
            v_loss = expectile_v_loss(
                stitched_target - v_all,
                float(self.cfg.expectile_tau),
            )
            self.v_optim.zero_grad(set_to_none=True)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.v.parameters(), float(self.cfg.grad_clip_norm))
            self.v_optim.step()
            v_head_losses.append(v_loss.detach())
            v_predictions.append(v_all.detach())
        else:
            for head_index, (head, optim) in enumerate(zip(self.v.heads, self.v_optims)):
                prediction = head(step_batch.v_state_feature)
                head_target = stitched_target[:, head_index : head_index + 1]
                head_loss = expectile_v_loss(
                    head_target - prediction,
                    float(self.cfg.expectile_tau),
                )
                optim.zero_grad(set_to_none=True)
                head_loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), float(self.cfg.grad_clip_norm))
                optim.step()
                v_head_losses.append(head_loss.detach())
                v_predictions.append(prediction.detach())
        self._polyak_update()

        with torch.no_grad():
            v_all_detached = torch.cat(v_predictions, dim=-1)
            v_mean = self._head_mean(v_all_detached)
            v_std = v_all_detached.std(dim=-1, unbiased=False).mean()
            stitched_target_mean = self._head_mean(stitched_target)
            stitched_advantage = stitched_target_mean - v_mean
            v_loss_values = torch.stack(v_head_losses)
            v_loss = v_loss_values.mean()
            mc_active = mc_mask.sum().clamp_min(1.0)
            comp_active = comp_mask.sum().clamp_min(1.0)
            mc_abs = ((g_tk.detach() - returns).abs() * mc_mask).sum() / mc_active
            comp_abs = comp_error.detach().abs().sum() / comp_active

        return {
            "g_loss": float(g_loss.detach().item()),
            "g_mc_loss": float(g_mc_loss.detach().item()),
            "g_comp_loss": float(g_comp_loss.detach().item()),
            "g_mc_error_abs_mean": float(mc_abs.item()),
            "g_comp_residual_abs_mean": float(comp_abs.item()),
            "v_loss": float(v_loss.detach().item()),
            "v_head_loss_mean": float(v_loss_values.mean().item()),
            "v_head_loss_min": float(v_loss_values.min().item()),
            "v_head_loss_max": float(v_loss_values.max().item()),
            "v_mean": float(v_mean.mean().item()),
            "v_std_mean": float(v_std.item()),
            "target_mean": float(stitched_target_mean.mean().item()),
            "stitched_target_mean": float(stitched_target_mean.mean().item()),
            "stitched_advantage_mean": float(stitched_advantage.mean().item()),
            "td_error_abs_mean": float(stitched_advantage.abs().mean().item()),
            "k_mean": float(k.float().mean().item()),
            "mc_mask_mean": float(mc_mask.mean().item()),
            "comp_mask_mean": float(comp_mask.mean().item()),
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
        """TD residual using the single head or independent-ensemble mean V.

        Shapes: features (B, D_state); rewards/dones (B, 1) or (B,). Returns
        (B,). This is the same backup V is trained on (:meth:`update`), read out
        on the scalar V readout, so the residual measures how much
        better-than-baseline the transition is.
        """
        bootstrap_discount = float(self.cfg.discount) ** int(self.cfg.action_horizon)
        v_s = self.v_value(v_state_feature)
        v_sp = self.target_v_value(next_v_state_feature)
        rewards = rewards.reshape_as(v_s)
        dones = dones.reshape_as(v_s)
        advantage = rewards + bootstrap_discount * (1.0 - dones) * v_sp - v_s
        return advantage.reshape(-1)

    @torch.no_grad()
    def compute_stitched_advantage(
        self,
        v_state_feature: torch.Tensor,
        future_v_state_feature: torch.Tensor,
        k: torch.Tensor,
        future_dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``G(s,s_k,k)+gamma^(kH)V_target(s_k)-V(s)`` as (B,).

        ``future_dones`` denotes whether an absorbing terminal was encountered
        anywhere along the selected macro path. Omitting it is equivalent to a
        non-terminal path and is intended for prevalidated Phase-B windows.
        """
        g_tk = self.g_value(v_state_feature, future_v_state_feature, k)
        v_future = self.target_v_value(future_v_state_feature)
        v_current = self.v_value(v_state_feature)
        dones = (
            torch.zeros_like(g_tk)
            if future_dones is None
            else future_dones.to(g_tk.device, g_tk.dtype).reshape_as(g_tk)
        )
        discount = torch.pow(
            torch.full_like(g_tk, float(self.cfg.discount)),
            k.to(g_tk.device, g_tk.dtype).reshape_as(g_tk)
            * int(self.cfg.action_horizon),
        )
        return (g_tk + discount * (1.0 - dones) * v_future - v_current).reshape(-1)

    # Alias retained for callers that use the method name from the design doc.
    compute_vast_advantage = compute_stitched_advantage

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "v": self.v.state_dict(),
            "target_v": self.target_v.state_dict(),
            "cfg": asdict(self.cfg),
            "algorithm": "vast_value_stitching_adaptation",
            "vast_v_mode": self.vast_v_mode,
            "learner_schema_version": 7 if self.vast_v_mode == "single_vast" else 8,
            "state_feature_dim": self.state_feature_dim,
            "chunk_feature_dim": self.chunk_feature_dim,
            "action_dim": self.action_dim,
            "n_tokens": self.n_tokens,
            "proprio_dim": self.proprio_dim,
            "state_proj_dim": self.state_proj_dim,
            "proprio_proj_dim": self.proprio_proj_dim,
            "v_ensemble_size": self.ensemble_size,
            "vast_max_k": int(self.cfg.vast_max_k),
            "action_horizon": int(self.cfg.action_horizon),
            "vast_sampling_seed": int(self.cfg.vast_sampling_seed),
            "g": self.g.state_dict(),
            "g_optim": self.g_optim.state_dict(),
        }
        if self.vast_v_mode == "single_vast":
            assert self.v_optim is not None
            state["v_optim"] = self.v_optim.state_dict()
        else:
            state["v_optims"] = [optim.state_dict() for optim in self.v_optims]
            state["ensemble_method"] = "independent_v_mean"
            state["ensemble_optimizer_count"] = len(self.v_optims)
        return state

    def load_state_dict(self, sd: dict[str, Any], strict: bool = True) -> None:
        if "q_ensemble" in sd or "q1" in sd:
            raise ValueError(
                "VASTLearner.load_state_dict: checkpoint contains an unsupported "
                "Q head (q_ensemble/q1). Re-run the VAST warmup."
            )
        schema = int(sd.get("learner_schema_version", -1))
        checkpoint_mode = str(sd.get("vast_v_mode", ""))
        if checkpoint_mode == "ensemble_lcb":
            raise ValueError(
                "VASTLearner.load_state_dict: legacy ensemble_lcb checkpoints are "
                "incompatible with indep_ensemble; re-run VAST warmup."
            )
        if schema not in {6, 7, 8}:
            raise ValueError(
                "VASTLearner.load_state_dict: unsupported learner schema "
                f"{schema}; expected schema 6, 7, or 8."
            )
        if "state_feature_dim" not in sd or "v" not in sd:
            raise ValueError(
                "VASTLearner.load_state_dict: checkpoint is missing required "
                "VAST state metadata."
            )
        if schema in {7, 8}:
            algorithm = str(sd.get("algorithm", ""))
            if algorithm != "vast_value_stitching_adaptation":
                raise ValueError(
                    f"VASTLearner.load_state_dict: schema-{schema} checkpoint has "
                    f"algorithm={algorithm!r}; expected "
                    "'vast_value_stitching_adaptation'."
                )
        else:
            checkpoint_method = str(sd.get("method", ""))
            if checkpoint_method != "vast_value_stitching":
                raise ValueError(
                    "VASTLearner.load_state_dict: schema-6 checkpoint is not "
                    "a VAST value-stitching state."
                )
            warnings.warn(
                "Loading legacy VAST learner schema 6; re-save as schema 7.",
                FutureWarning,
                stacklevel=2,
            )
        if schema in {6, 7} and checkpoint_mode not in {"", "single_vast"}:
            raise ValueError(
                "VASTLearner.load_state_dict: pre-schema-8 ensemble checkpoints are "
                "unsupported; re-run VAST warmup with vast_v_mode=indep_ensemble."
            )
        if schema == 8:
            if checkpoint_mode != "indep_ensemble":
                raise ValueError(
                    "VASTLearner.load_state_dict: schema-8 is reserved for "
                    f"indep_ensemble, got vast_v_mode={checkpoint_mode!r}."
                )
            if sd.get("ensemble_method") != "independent_v_mean":
                raise ValueError(
                    "VASTLearner.load_state_dict: schema-8 checkpoint has invalid "
                    f"ensemble_method={sd.get('ensemble_method')!r}."
                )
        if "g" not in sd or "g_optim" not in sd:
            raise ValueError(
                "VASTLearner.load_state_dict: VAST checkpoint has no G/g_optim state. "
                "Re-run offline VAST warmup."
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
                        f"VASTLearner.load_state_dict: {key} mismatch "
                        f"(ckpt={sd.get(key)}, runtime={runtime_val})"
                    )
            if int(sd["state_feature_dim"]) != self.state_feature_dim:
                raise ValueError(
                    "VASTLearner.load_state_dict: state_feature_dim mismatch "
                    f"(ckpt={sd['state_feature_dim']}, runtime={self.state_feature_dim})"
                )
            for key, runtime_val in (
                ("vast_v_mode", self.vast_v_mode),
                ("vast_max_k", int(self.cfg.vast_max_k)),
                ("vast_sampling_seed", int(self.cfg.vast_sampling_seed)),
                ("action_horizon", int(self.cfg.action_horizon)),
            ):
                checkpoint_val = sd.get(key)
                if checkpoint_val != runtime_val:
                    raise ValueError(
                        f"VASTLearner.load_state_dict: {key} mismatch "
                        f"(ckpt={checkpoint_val!r}, runtime={runtime_val!r})"
                    )
            saved_cfg = sd.get("cfg", {})
            for key in (
                "discount",
                "expectile_tau",
                "vast_comp_coef",
                "output_reward_coef",
                "disc_reward_coef",
            ):
                if float(saved_cfg.get(key, float("nan"))) != float(getattr(self.cfg, key)):
                    raise ValueError(
                        f"VASTLearner.load_state_dict: config {key} mismatch "
                        f"(ckpt={saved_cfg.get(key)!r}, runtime={getattr(self.cfg, key)!r})"
                    )
        self.v.load_state_dict(sd["v"])
        self.target_v.load_state_dict(sd["target_v"])
        if self.vast_v_mode == "single_vast":
            assert self.v_optim is not None
            self.v_optim.load_state_dict(sd["v_optim"])
        else:
            if int(sd.get("ensemble_optimizer_count", -1)) != self.ensemble_size:
                raise ValueError(
                    "VASTLearner.load_state_dict: ensemble_optimizer_count mismatch "
                    f"(ckpt={sd.get('ensemble_optimizer_count')}, "
                    f"runtime={self.ensemble_size})."
                )
            saved_optims = sd.get("v_optims")
            if not isinstance(saved_optims, list) or len(saved_optims) != len(self.v_optims):
                raise ValueError(
                    "VASTLearner.load_state_dict: ensemble optimizer count mismatch "
                    f"(ckpt={0 if not isinstance(saved_optims, list) else len(saved_optims)}, "
                    f"runtime={len(self.v_optims)})."
                )
            for optim, saved_optim in zip(self.v_optims, saved_optims):
                optim.load_state_dict(saved_optim)
        self.g.load_state_dict(sd["g"])
        self.g_optim.load_state_dict(sd["g_optim"])
