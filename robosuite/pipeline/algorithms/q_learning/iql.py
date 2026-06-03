"""IQL learner with independent frozen ResNet-50 Q/V visual encoders."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .common import IQLActorBatch, IQLConfig, IQLStepBatch
from .losses import bellman_q_loss, compute_advantage, expectile_v_loss
from .networks import (
    QChunkNetwork,
    VNetwork,
    resolved_resnet_path,
    trainable_parameters,
)


IQL_STATE_SCHEMA_VERSION = 2


class IQLLearner:
    """Q-chunking IQL over raw multi-view images, proprio and action chunks."""

    def __init__(
        self,
        cfg: IQLConfig,
        *,
        camera_names: list[str] | tuple[str, ...],
        proprio_dim: int,
        action_dim: int,
    ) -> None:
        self.cfg = cfg
        self.camera_names = tuple(str(name) for name in camera_names)
        if not self.camera_names:
            raise ValueError("IQLLearner requires at least one camera name")
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.resnet_pretrained_path = resolved_resnet_path(cfg.resnet_pretrained_path)
        device = cfg.device

        q_kwargs = {
            "num_cameras": len(self.camera_names),
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "action_horizon": int(cfg.action_horizon),
            "resnet_pretrained_path": self.resnet_pretrained_path,
            "state_feature_dim": int(cfg.state_feature_dim),
            "action_feature_dim": int(cfg.action_feature_dim),
            "hidden_dims": tuple(cfg.hidden_dims),
        }
        v_kwargs = {
            "num_cameras": len(self.camera_names),
            "proprio_dim": self.proprio_dim,
            "resnet_pretrained_path": self.resnet_pretrained_path,
            "state_feature_dim": int(cfg.state_feature_dim),
            "hidden_dims": tuple(cfg.hidden_dims),
        }
        self.q1: QChunkNetwork = QChunkNetwork(**q_kwargs).to(device)
        self.q2: QChunkNetwork = QChunkNetwork(**q_kwargs).to(device)
        self.v: VNetwork = VNetwork(**v_kwargs).to(device)
        self.target_v: VNetwork = VNetwork(**v_kwargs).to(device)
        self.target_v.load_state_dict(self.v.state_dict())
        for param in self.target_v.parameters():
            param.requires_grad_(False)

        self._q_trainable = trainable_parameters(self.q1) + trainable_parameters(self.q2)
        self._v_trainable = trainable_parameters(self.v)
        self.q_optim: torch.optim.Optimizer = torch.optim.AdamW(
            self._q_trainable,
            lr=float(cfg.q_lr),
            weight_decay=float(cfg.weight_decay),
        )
        self.v_optim: torch.optim.Optimizer = torch.optim.AdamW(
            self._v_trainable,
            lr=float(cfg.v_lr),
            weight_decay=float(cfg.weight_decay),
        )

    def _model_meta(self) -> dict[str, Any]:
        return {
            "camera_names": list(self.camera_names),
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "action_horizon": int(self.cfg.action_horizon),
            "resnet_pretrained_path": self.resnet_pretrained_path,
            "state_feature_dim": int(self.cfg.state_feature_dim),
            "action_feature_dim": int(self.cfg.action_feature_dim),
        }

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _bootstrap_target(self, step_batch: IQLStepBatch) -> torch.Tensor:
        """Bellman target r + gamma^H * (1 - done) * target_v(s')."""
        bootstrap_discount = float(self.cfg.discount) ** int(self.cfg.action_horizon)
        with torch.no_grad():
            v_next = self.target_v(
                step_batch.next_image_obs_raw,
                step_batch.next_proprio_raw,
            )
            target = step_batch.rewards + bootstrap_discount * (1.0 - step_batch.dones) * v_next
        return target

    @torch.no_grad()
    def _polyak_update(self) -> None:
        tau_p = float(self.cfg.target_polyak)
        target_params = dict(self.target_v.named_parameters())
        for name, source in self.v.named_parameters():
            if source.requires_grad:
                target = target_params[name]
                target.data.mul_(1.0 - tau_p).add_(source.data, alpha=tau_p)

    # ------------------------------------------------------------------ #
    # Training                                                           #
    # ------------------------------------------------------------------ #

    def update(self, step_batch: IQLStepBatch) -> dict[str, float]:
        target_q = self._bootstrap_target(step_batch)

        q1_pred = self.q1(
            step_batch.image_obs_raw,
            step_batch.proprio_raw,
            step_batch.action_chunk,
        )
        q2_pred = self.q2(
            step_batch.image_obs_raw,
            step_batch.proprio_raw,
            step_batch.action_chunk,
        )
        q_loss = bellman_q_loss(q1_pred, target_q) + bellman_q_loss(q2_pred, target_q)
        self.q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._q_trainable, float(self.cfg.grad_clip_norm))
        self.q_optim.step()

        with torch.no_grad():
            q_min = torch.min(q1_pred.detach(), q2_pred.detach())

        v_pred = self.v(step_batch.image_obs_raw, step_batch.proprio_raw)
        diff = q_min - v_pred
        v_loss = expectile_v_loss(diff, float(self.cfg.expectile_tau))
        self.v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._v_trainable, float(self.cfg.grad_clip_norm))
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
        target = self._bootstrap_target(step_batch)
        v_pred = self.v(step_batch.image_obs_raw, step_batch.proprio_raw)
        v_loss = torch.nn.functional.mse_loss(v_pred, target)
        self.v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._v_trainable, float(self.cfg.grad_clip_norm))
        self.v_optim.step()
        self._polyak_update()
        return {
            "v_loss": float(v_loss.detach().item()),
            "v_mean": float(v_pred.detach().mean().item()),
            "target_mean": float(target.detach().mean().item()),
        }

    # ------------------------------------------------------------------ #
    # Advantage scoring                                                  #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def compute_advantage_for_batch(self, actor_batch: IQLActorBatch) -> torch.Tensor:
        q1 = self.q1(
            actor_batch.image_obs_raw,
            actor_batch.proprio_raw,
            actor_batch.action_chunk_raw,
        )
        q2 = self.q2(
            actor_batch.image_obs_raw,
            actor_batch.proprio_raw,
            actor_batch.action_chunk_raw,
        )
        v = self.v(actor_batch.image_obs_raw, actor_batch.proprio_raw)
        return compute_advantage(q1, q2, v)

    # ------------------------------------------------------------------ #
    # Persistence                                                        #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": IQL_STATE_SCHEMA_VERSION,
            "q1": self.q1.compact_state_dict(),
            "q2": self.q2.compact_state_dict(),
            "v": self.v.compact_state_dict(),
            "target_v": self.target_v.compact_state_dict(),
            "q_optim": self.q_optim.state_dict(),
            "v_optim": self.v_optim.state_dict(),
            "cfg": asdict(self.cfg),
            "model_meta": self._model_meta(),
        }

    def load_state_dict(self, state: dict[str, Any], strict: bool = True) -> None:
        schema_version = int(state.get("schema_version", -1))
        if schema_version != IQL_STATE_SCHEMA_VERSION:
            raise ValueError(
                "IQLLearner.load_state_dict requires schema_version=2 ResNet-50 state; "
                f"got {state.get('schema_version')!r}. Re-run Q/V warmup."
            )
        if strict:
            checkpoint_meta = dict(state.get("model_meta", {}))
            runtime_meta = self._model_meta()
            for key, runtime_value in runtime_meta.items():
                if checkpoint_meta.get(key) != runtime_value:
                    raise ValueError(
                        f"IQLLearner.load_state_dict: {key} mismatch "
                        f"(ckpt={checkpoint_meta.get(key)!r}, runtime={runtime_value!r})"
                    )
        self.q1.load_compact_state_dict(state["q1"])
        self.q2.load_compact_state_dict(state["q2"])
        self.v.load_compact_state_dict(state["v"])
        self.target_v.load_compact_state_dict(state["target_v"])
        self.q_optim.load_state_dict(state["q_optim"])
        self.v_optim.load_state_dict(state["v_optim"])


__all__ = ["IQLLearner", "IQL_STATE_SCHEMA_VERSION"]
