"""Chunk-centric replay sampler for IQL.

Wraps the same underlying `Transition` store used by `DipoleReplayBuffer`
(any `FlowDaggerReplayBuffer` subclass) to avoid duplicating frames. On
sample it:
  1. Picks valid windows where all H steps share the same episode (delegates
     to the base buffer's valid-start cache).
  2. Validates the `start + H` boundary for `s'`; falls back to
     `_storage[start+H-1].next_obs` (and forces done=1) if the next step is
     out of storage or cross-episode.
  3. Runs the frozen `SharedFrozenEncoder` on s and s' under `no_grad`.
  4. Synthesizes `r_total = r_env + disc_reward_coef · r_disc` using the
     CURRENT discriminator (re-evaluated per sample to avoid stale rewards).
     When `discriminator is None` the disc term is skipped without mutating
     `cfg.disc_reward_coef`.
  5. Returns `IQLStepBatch` / `IQLActorBatch`.

Mirrors baseline awr/replay_buffer.py:295-510 for the windowing logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from robosuite.pipeline.algorithms.flow_dagger.replay_buffer import (
    _center_crop_resize,
)

from .common import IQLActorBatch, IQLConfig, IQLStepBatch
from .data_util import aggregate_chunk_reward, chunk_done_mask

if TYPE_CHECKING:
    from robosuite.pipeline.algorithms.discriminator.encoder import SharedFrozenEncoder
    from robosuite.pipeline.algorithms.discriminator.online_bce import (
        OnlineBCEDiscriminator,
    )


def _episode_index_of(transition: Any) -> int:
    info = getattr(transition, "info", None) or {}
    return int(info.get("episode_index", -1))


def _stack_views_uint8(obs: Any, camera_names: list[str], image_size: int) -> np.ndarray:
    images = []
    for camera_name in camera_names:
        if camera_name not in obs:
            raise KeyError(f"IQL replay sample is missing camera '{camera_name}'")
        image = np.asarray(obs[camera_name], dtype=np.uint8)
        image = _center_crop_resize(image, image_size)
        images.append(np.transpose(image, (2, 0, 1)))  # (3, H, W)
    return np.stack(images, axis=0)  # (V, 3, H, W)


def _to_image_tensor(stacked: np.ndarray, device: str) -> torch.Tensor:
    """(B, V, 3, H, W) uint8 -> float32 in [0, 1] on cpu (encoder moves to its own device)."""
    tensor = torch.from_numpy(np.ascontiguousarray(stacked)).to(dtype=torch.float32)
    return tensor.div_(255.0)


class IQLReplayBuffer:
    """Chunk-centric replay buffer wrapping a `FlowDaggerReplayBuffer`-style store.

    Args:
        base_buffer: the underlying transition store (typically the DIPOLE
            online buffer; passed by reference, not copied). Must expose
            `_storage`, `_lock`, `_get_valid_start_indices_locked()`,
            `camera_names`, `image_size`, `action_horizon`.
        cfg: IQLConfig (carries action_horizon, discount, disc_reward_coef).
    """

    def __init__(self, base_buffer: Any, cfg: IQLConfig) -> None:
        self._base = base_buffer
        self.cfg = cfg
        if int(getattr(base_buffer, "action_horizon", -1)) != int(cfg.action_horizon):
            raise ValueError(
                "IQLReplayBuffer: base_buffer.action_horizon "
                f"({getattr(base_buffer, 'action_horizon', None)}) must equal "
                f"cfg.action_horizon ({cfg.action_horizon}); chunk-window cache "
                "would otherwise be inconsistent."
            )

    # ------------------------------------------------------------------ #
    # Sampling                                                            #
    # ------------------------------------------------------------------ #

    def _gather_chunks(self, batch_size: int) -> tuple[list[list[Any]], list[int]]:
        """Sample `batch_size` chunk sequences from the base buffer. Returns
        `(sequences, start_indices)` — sequences is a list of H-length lists
        of `Transition` objects (held by reference; do not mutate)."""
        with self._base._lock:  # noqa: SLF001 — intentional access to base cache
            valid_starts = self._base._get_valid_start_indices_locked()  # noqa: SLF001
            if len(valid_starts) == 0:
                raise ValueError("IQLReplayBuffer: base buffer has no valid sequences.")
            sampled = np.random.randint(0, len(valid_starts), size=int(batch_size))
            start_indices = [valid_starts[int(i)] for i in sampled]
            H = int(self.cfg.action_horizon)
            sequences = [self._base._storage[s : s + H] for s in start_indices]  # noqa: SLF001
        return sequences, start_indices

    def _next_obs_for(self, start: int) -> tuple[Any, bool]:
        """Resolve s' for chunk starting at `start`. Returns (next_obs, forced_done)."""
        H = int(self.cfg.action_horizon)
        with self._base._lock:  # noqa: SLF001
            storage = self._base._storage  # noqa: SLF001
            n = len(storage)
            first_episode = _episode_index_of(storage[start])
            tail_idx = start + H
            if tail_idx < n:
                tail_episode = _episode_index_of(storage[tail_idx])
                if tail_episode == first_episode:
                    return storage[tail_idx].obs, False
            return storage[start + H - 1].next_obs, True

    def sample_step_batch(
        self,
        batch_size: int,
        *,
        encoder: "SharedFrozenEncoder",
        discriminator: "OnlineBCEDiscriminator | None" = None,
        device: str = "cuda:1",
    ) -> IQLStepBatch:
        """Sample `batch_size` chunk windows, encode contexts under no_grad,
        and synthesize total rewards (env + optional disc intrinsic).
        """
        sequences, start_indices = self._gather_chunks(batch_size)
        camera_names = list(self._base.camera_names)
        image_size = int(self._base.image_size)
        H = int(self.cfg.action_horizon)

        s_images: list[np.ndarray] = []
        sp_images: list[np.ndarray] = []
        s_proprio: list[np.ndarray] = []
        sp_proprio: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards_per_step: list[np.ndarray] = []
        dones_per_step: list[np.ndarray] = []
        is_online: list[float] = []
        is_intervention: list[float] = []
        episode_ids: list[int] = []
        episode_steps: list[int] = []

        for sequence, start in zip(sequences, start_indices):
            first = sequence[0]
            s_obs = first.obs
            next_obs, forced_done = self._next_obs_for(start)
            s_images.append(_stack_views_uint8(s_obs, camera_names, image_size))
            sp_images.append(_stack_views_uint8(next_obs, camera_names, image_size))
            s_proprio.append(np.asarray(s_obs["state"], dtype=np.float32))
            sp_proprio.append(np.asarray(next_obs["state"], dtype=np.float32))
            actions.append(
                np.stack([np.asarray(item.action, dtype=np.float32) for item in sequence], axis=0)
            )
            step_rewards = np.asarray(
                [float(item.reward) if item.reward is not None else 0.0 for item in sequence],
                dtype=np.float32,
            )
            step_dones = np.asarray([bool(item.done) for item in sequence], dtype=np.float32)
            if forced_done:
                step_dones[-1] = 1.0
            rewards_per_step.append(step_rewards)
            dones_per_step.append(step_dones)
            info = first.info or {}
            episode_ids.append(int(info.get("episode_index", -1)))
            episode_steps.append(int(info.get("episode_step", -1)))
            buffer_role = str(info.get("buffer_role", "offline")).lower()
            is_online.append(1.0 if buffer_role == "online" else 0.0)
            is_intervention.append(1.0 if bool(first.is_intervention) else 0.0)

        s_image_tensor = _to_image_tensor(np.stack(s_images, axis=0), device)
        sp_image_tensor = _to_image_tensor(np.stack(sp_images, axis=0), device)
        s_proprio_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(s_proprio, axis=0))).float()
        sp_proprio_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(sp_proprio, axis=0))).float()
        action_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(actions, axis=0))).float()
        reward_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(rewards_per_step, axis=0))).float()
        done_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(dones_per_step, axis=0))).float()
        is_online_tensor = torch.tensor(is_online, dtype=torch.float32).unsqueeze(-1)
        is_intervention_tensor = torch.tensor(is_intervention, dtype=torch.float32).unsqueeze(-1)

        with torch.no_grad():
            context = encoder.encode(image_obs_raw=s_image_tensor, proprio_raw=s_proprio_tensor)
            next_context = encoder.encode(image_obs_raw=sp_image_tensor, proprio_raw=sp_proprio_tensor)

        # Disc intrinsic reward: applied once per chunk (NOT H times), then
        # broadcast across the H steps by dividing by H so the n-step aggregator
        # treats it as a per-step contribution. Skipped entirely when no disc
        # is provided so warmup can run before subagent-2 lands.
        #
        # Sign convention (matches lpb_v2 BCE warm-start): higher logit =
        # more failure-like. `disc_reward_sign="negate_logit"` (default)
        # therefore turns intrinsic_reward into r_disc = -logit so the
        # agent is *rewarded* for being non-failure-like.
        effective_disc_coef = 0.0 if discriminator is None else float(self.cfg.disc_reward_coef)
        if effective_disc_coef != 0.0 and discriminator is not None:
            action_for_disc = action_tensor.to(context.device, dtype=context.dtype)
            with torch.no_grad():
                r_disc_chunk = discriminator.intrinsic_reward(
                    context=context, action_chunk=action_for_disc
                )  # (B,) — raw logit, higher = more failure-like.
            sign_mode = str(self.cfg.disc_reward_sign).lower()
            if sign_mode == "negate_logit":
                r_disc_chunk = -r_disc_chunk
            elif sign_mode == "raw":
                pass
            else:
                raise ValueError(
                    f"IQLConfig.disc_reward_sign must be 'negate_logit' or "
                    f"'raw'; got {self.cfg.disc_reward_sign!r}"
                )
            r_disc_per_step = (r_disc_chunk.unsqueeze(-1) / float(H)).expand(-1, H)
            r_disc_per_step = r_disc_per_step.to(reward_tensor.device, dtype=reward_tensor.dtype)
        else:
            r_disc_per_step = torch.zeros_like(reward_tensor)

        r_total_chunk = reward_tensor + effective_disc_coef * r_disc_per_step
        rewards = aggregate_chunk_reward(r_total_chunk, float(self.cfg.discount))
        dones = chunk_done_mask(done_tensor)

        batch = IQLStepBatch(
            context=context,
            next_context=next_context,
            action_chunk=action_tensor,
            rewards=rewards,
            dones=dones,
            is_online=is_online_tensor,
            is_intervention=is_intervention_tensor,
            metadata={
                "start_indices": start_indices,
                "episode_ids": episode_ids,
                "episode_steps": episode_steps,
            },
        )
        return batch.to(device)

    def sample_actor_batch(
        self,
        batch_size: int,
        *,
        encoder: "SharedFrozenEncoder",
        device: str = "cuda:1",
    ) -> IQLActorBatch:
        """Sample chunks for actor-side advantage scoring (no rewards needed)."""
        sequences, start_indices = self._gather_chunks(batch_size)
        camera_names = list(self._base.camera_names)
        image_size = int(self._base.image_size)

        s_images: list[np.ndarray] = []
        s_proprio: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        episode_ids: list[int] = []
        episode_steps: list[int] = []
        for sequence in sequences:
            first = sequence[0]
            s_obs = first.obs
            s_images.append(_stack_views_uint8(s_obs, camera_names, image_size))
            s_proprio.append(np.asarray(s_obs["state"], dtype=np.float32))
            actions.append(
                np.stack([np.asarray(item.action, dtype=np.float32) for item in sequence], axis=0)
            )
            info = first.info or {}
            episode_ids.append(int(info.get("episode_index", -1)))
            episode_steps.append(int(info.get("episode_step", -1)))

        s_image_tensor = _to_image_tensor(np.stack(s_images, axis=0), device)
        s_proprio_tensor = torch.from_numpy(np.ascontiguousarray(np.stack(s_proprio, axis=0))).float()
        action_tensor_raw = torch.from_numpy(np.ascontiguousarray(np.stack(actions, axis=0))).float()

        with torch.no_grad():
            context = encoder.encode(image_obs_raw=s_image_tensor, proprio_raw=s_proprio_tensor)

        batch = IQLActorBatch(
            context=context,
            action_chunk_raw=action_tensor_raw,
            metadata={
                "start_indices": start_indices,
                "episode_ids": episode_ids,
                "episode_steps": episode_steps,
            },
        )
        return batch.to(device)

    # ------------------------------------------------------------------ #
    # Capacity / readiness                                                #
    # ------------------------------------------------------------------ #

    def ready(self, batch_size: int) -> bool:
        """True iff enough valid windows exist to sample `batch_size`."""
        with self._base._lock:  # noqa: SLF001
            valid_starts = self._base._get_valid_start_indices_locked()  # noqa: SLF001
        return len(valid_starts) >= int(batch_size)

    def __len__(self) -> int:
        return len(self._base)
