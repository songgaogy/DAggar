from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from robosuite.pipeline.algorithms.discriminator.lpb_v2_scorer import (
    lpb_disc_intrinsic_from_failure_score,
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


def _is_truncated_boundary(transition: Any) -> bool:
    info = getattr(transition, "info", None) or {}
    return bool(info.get("is_truncated_boundary", False)) or (
        str(info.get("episode_terminal_reason", "")).lower() == "truncated"
    )


def _lpb_disc_intrinsic_for_transition(transition: Any) -> float | None:
    """Read precomputed LPB disc reward from ``transition.info`` (warmup / offline)."""
    info = getattr(transition, "info", None) or {}
    cached = info.get("lpb_disc_intrinsic")
    if cached is not None:
        return float(cached)
    failure_score = info.get("lpb_failure_score")
    tau = info.get("lpb_tau")
    if failure_score is None or tau is None:
        return None
    return float(lpb_disc_intrinsic_from_failure_score(float(failure_score), float(tau)))


def _lpb_disc_steps_for_sequence(sequence: list[Any], horizon: int) -> np.ndarray | None:
    steps: list[float] = []
    for item in sequence:
        intrinsic = _lpb_disc_intrinsic_for_transition(item)
        if intrinsic is None:
            return None
        steps.append(intrinsic)
    if len(steps) != int(horizon):
        return None
    return np.asarray(steps, dtype=np.float32)


def _stack_views_uint8(obs: Any, camera_names: list[str]) -> np.ndarray:
    """Stack per-camera uint8 frames from ``obs`` into (V, 3, H, W).

    No resize / crop: the SharedFrozenEncoder owns the full LPB v2
    preprocessing pipeline (F.interpolate to ``original_img_size`` →
    LinearNormalizer → CenterCrop to ``cropped_img_size``). Doing any
    of that in the replay sampler would double-process the input.
    See DEBUG_and_ERRORs.md §4.
    """
    images = []
    for camera_name in camera_names:
        if camera_name not in obs:
            raise KeyError(f"IQL replay sample is missing camera '{camera_name}'")
        image = np.asarray(obs[camera_name], dtype=np.uint8)
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
        cfg: IQLConfig
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

    def _has_bootstrap_or_true_terminal_locked(self, start: int) -> bool:
        H = int(self.cfg.action_horizon)
        storage = self._base._storage  # noqa: SLF001
        n = len(storage)
        first_episode = _episode_index_of(storage[start])
        tail_idx = int(start) + H
        if tail_idx < n and _episode_index_of(storage[tail_idx]) == first_episode:
            return True
        last = storage[start + H - 1]
        return bool(last.done) and not _is_truncated_boundary(last)

    def _get_iql_valid_start_indices_locked(self) -> list[int]:
        valid_starts = self._base._get_valid_start_indices_locked()  # noqa: SLF001
        return [
            int(start)
            for start in valid_starts
            if self._has_bootstrap_or_true_terminal_locked(int(start))
        ]

    def _sample_start_indices(self, batch_size: int) -> list[int]:
        """Sample valid chunk start indices from the base buffer."""
        with self._base._lock:  # noqa: SLF001 — intentional access to base cache
            valid_starts = self._get_iql_valid_start_indices_locked()
            if len(valid_starts) == 0:
                raise ValueError("IQLReplayBuffer: base buffer has no valid sequences.")
            sampled = np.random.randint(0, len(valid_starts), size=int(batch_size))
            return [valid_starts[int(i)] for i in sampled]

    def _gather_chunks_for_starts(self, start_indices: list[int]) -> list[list[Any]]:
        """Gather H-step chunk sequences for explicit start indices."""
        with self._base._lock:  # noqa: SLF001 — intentional access to base cache
            H = int(self.cfg.action_horizon)
            sequences = [self._base._storage[s : s + H] for s in start_indices]  # noqa: SLF001
        return sequences

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
            last = storage[start + H - 1]
            if bool(last.done) and not _is_truncated_boundary(last):
                return last.next_obs, True
            raise ValueError(
                "IQLReplayBuffer: requested a truncated boundary chunk without "
                f"bootstrap next_obs (start={start}, horizon={H})."
            )

    def _build_step_batch_from_start_indices(
        self,
        start_indices: list[int],
        *,
        encoder: "SharedFrozenEncoder",
        discriminator: "OnlineBCEDiscriminator | None" = None,
        device: str = "cuda:1",
    ) -> IQLStepBatch:
        """Build an encoded IQL batch for explicit valid chunk starts."""
        if not start_indices:
            raise ValueError("IQLReplayBuffer: start_indices must be non-empty.")
        sequences = self._gather_chunks_for_starts(start_indices)
        camera_names = list(self._base.camera_names)
        H = int(self.cfg.action_horizon)

        chunk_images: list[np.ndarray] = []     # each (H, V, 3, h, w)
        chunk_proprio: list[np.ndarray] = []    # each (H, D_s)
        sp_images: list[np.ndarray] = []        # each (V, 3, h, w)
        sp_proprio: list[np.ndarray] = []       # each (D_s,)
        actions: list[np.ndarray] = []          # each (H, D_a)
        rewards_per_step: list[np.ndarray] = []
        dones_per_step: list[np.ndarray] = []
        is_online: list[float] = []
        is_intervention: list[float] = []
        episode_ids: list[int] = []
        episode_steps: list[int] = []
        lpb_disc_per_sequence: list[np.ndarray | None] = []

        for sequence, start in zip(sequences, start_indices):
            lpb_disc_per_sequence.append(_lpb_disc_steps_for_sequence(sequence, H))
            first = sequence[0]
            next_obs, forced_done = self._next_obs_for(start)
            chunk_images.append(
                np.stack(
                    [_stack_views_uint8(item.obs, camera_names) for item in sequence],
                    axis=0,
                )
            )
            chunk_proprio.append(
                np.stack(
                    [np.asarray(item.obs["state"], dtype=np.float32) for item in sequence],
                    axis=0,
                )
            )
            sp_images.append(_stack_views_uint8(next_obs, camera_names))
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

        B = len(sequences)
        chunk_images_np = np.stack(chunk_images, axis=0)            # (B, H, V, 3, h, w)
        chunk_proprio_np = np.stack(chunk_proprio, axis=0)          # (B, H, D_s)
        sp_images_np = np.stack(sp_images, axis=0)                  # (B, V, 3, h, w)
        sp_proprio_np = np.stack(sp_proprio, axis=0)                # (B, D_s)
        action_np = np.stack(actions, axis=0)                       # (B, H, D_a)
        rewards_np = np.stack(rewards_per_step, axis=0)             # (B, H)
        dones_np = np.stack(dones_per_step, axis=0)                 # (B, H)

        V, C, Hi, Wi = chunk_images_np.shape[2:]
        chunk_images_tensor = _to_image_tensor(
            np.ascontiguousarray(chunk_images_np.reshape(B * H, V, C, Hi, Wi)),
            device,
        ).view(B, H, V, C, Hi, Wi)
        chunk_proprio_tensor = torch.from_numpy(np.ascontiguousarray(chunk_proprio_np)).float()
        action_tensor = torch.from_numpy(np.ascontiguousarray(action_np)).float()
        reward_tensor = torch.from_numpy(np.ascontiguousarray(rewards_np)).float()
        done_tensor = torch.from_numpy(np.ascontiguousarray(dones_np)).float()
        is_online_tensor = torch.tensor(is_online, dtype=torch.float32).unsqueeze(-1)
        is_intervention_tensor = torch.tensor(is_intervention, dtype=torch.float32).unsqueeze(-1)

        sp_image_tensor = _to_image_tensor(sp_images_np, device)                # (B, V, 3, Hi, Wi)
        sp_proprio_tensor = torch.from_numpy(
            np.ascontiguousarray(sp_proprio_np)
        ).float()
        # next_obs: use the chunk's last action (the step that led to s').
        sp_action_tensor = action_tensor[:, -1, :]

        with torch.no_grad():
            chunk_ctx = encoder.encode_chunk_frames(
                chunk_images=chunk_images_tensor,
                chunk_proprio=chunk_proprio_tensor,
                chunk_actions=action_tensor,
            )                                                                   # (B, H, D_ctx)
            chunk_ctx_flat = chunk_ctx.reshape(B * H, -1)
            next_context = encoder.encode(
                image_obs_raw=sp_image_tensor,
                proprio_raw=sp_proprio_tensor,
                action_real=sp_action_tensor,
            )                                                                   # (B, D_ctx)

        D_ctx = int(chunk_ctx.shape[-1])
        # Q/V Bellman uses the chunk-start latent; disc scores all H frames.
        context = chunk_ctx[:, 0, :]

        # Per-frame disc reward: LPB pre-annotations (offline) or online head.
        effective_disc_coef = float(self.cfg.disc_reward_coef)
        effective_output_coef = float(self.cfg.output_reward_coef)
        use_lpb_disc = (
            effective_disc_coef != 0.0
            and len(lpb_disc_per_sequence) == B
            and all(step is not None for step in lpb_disc_per_sequence)
        )
        if effective_disc_coef == 0.0:
            r_disc_per_step = torch.zeros_like(reward_tensor)
        elif use_lpb_disc:
            r_disc_per_step = torch.from_numpy(
                np.stack([step for step in lpb_disc_per_sequence if step is not None])
            ).to(device=reward_tensor.device, dtype=reward_tensor.dtype)
        elif discriminator is not None:
            with torch.no_grad():
                r_disc_flat = discriminator.intrinsic_reward(context=chunk_ctx_flat)
            r_disc_per_step = r_disc_flat.view(B, H).to(
                reward_tensor.device, dtype=reward_tensor.dtype
            )
        else:
            r_disc_per_step = torch.zeros_like(reward_tensor)

        r_total_chunk = effective_output_coef * reward_tensor + effective_disc_coef * r_disc_per_step
        rewards = aggregate_chunk_reward(r_total_chunk, float(self.cfg.discount))
        dones = chunk_done_mask(done_tensor)

        disc_meta: dict[str, float] = {}
        if effective_disc_coef != 0.0 and (use_lpb_disc or discriminator is not None):
            disc_meta["disc_reward_first_frame_mean"] = float(r_disc_per_step[:, 0].mean().item())
            disc_meta["disc_reward_chunk_mean"] = float(r_disc_per_step.mean().item())

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
                **disc_meta,
            },
        )
        return batch.to(device)

    def sample_step_batch(
        self,
        batch_size: int,
        *,
        encoder: "SharedFrozenEncoder",
        discriminator: "OnlineBCEDiscriminator | None" = None,
        device: str = "cuda:1",
    ) -> IQLStepBatch:
        """Sample `batch_size` chunk windows, encode every frame under no_grad,
        and synthesize total rewards (env + optional disc intrinsic).

        When transitions carry ``info['lpb_disc_intrinsic']`` (offline warmup
        LPB benchmark scores), those values are used for ``r_disc``. Otherwise
        falls back to ``OnlineBCEDiscriminator.intrinsic_reward`` on
        ``encode_chunk_frames`` latents.
        """
        start_indices = self._sample_start_indices(batch_size)
        return self._build_step_batch_from_start_indices(
            start_indices,
            encoder=encoder,
            discriminator=discriminator,
            device=device,
        )

    def preencode_step_cache(
        self,
        *,
        encoder: "SharedFrozenEncoder",
        discriminator: "OnlineBCEDiscriminator | None" = None,
        device: str = "cuda:1",
        encode_batch_size: int = 64,
        cache_device: str = "cpu",
        progress_desc: str | None = None,
    ) -> "IQLPreencodedReplayCache":
        """Precompute encoded IQL step tensors for all current valid starts.

        Intended for offline warmup only. Online learning should keep using
        dynamic replay sampling so current discriminator rewards are re-scored.
        """
        with self._base._lock:  # noqa: SLF001
            valid_starts = list(self._get_iql_valid_start_indices_locked())
        if not valid_starts:
            raise ValueError("IQLReplayBuffer: base buffer has no valid sequences to preencode.")

        encode_bs = max(1, int(encode_batch_size))
        iterator = range(0, len(valid_starts), encode_bs)
        if progress_desc:
            from tqdm import tqdm

            iterator = tqdm(
                iterator,
                total=(len(valid_starts) + encode_bs - 1) // encode_bs,
                desc=progress_desc,
            )

        parts: dict[str, list[torch.Tensor]] = {
            "context": [],
            "next_context": [],
            "action_chunk": [],
            "rewards": [],
            "dones": [],
            "is_online": [],
            "is_intervention": [],
        }
        for start in iterator:
            batch_starts = valid_starts[start : start + encode_bs]
            batch = self._build_step_batch_from_start_indices(
                batch_starts,
                encoder=encoder,
                discriminator=discriminator,
                device=device,
            ).to(cache_device)
            parts["context"].append(batch.context.detach())
            parts["next_context"].append(batch.next_context.detach())
            parts["action_chunk"].append(batch.action_chunk.detach())
            parts["rewards"].append(batch.rewards.detach())
            parts["dones"].append(batch.dones.detach())
            parts["is_online"].append(batch.is_online.detach())
            parts["is_intervention"].append(batch.is_intervention.detach())

        return IQLPreencodedReplayCache(
            context=torch.cat(parts["context"], dim=0),
            next_context=torch.cat(parts["next_context"], dim=0),
            action_chunk=torch.cat(parts["action_chunk"], dim=0),
            rewards=torch.cat(parts["rewards"], dim=0),
            dones=torch.cat(parts["dones"], dim=0),
            is_online=torch.cat(parts["is_online"], dim=0),
            is_intervention=torch.cat(parts["is_intervention"], dim=0),
            source_size=len(valid_starts),
        )

    def sample_actor_batch(
        self,
        batch_size: int,
        *,
        encoder: "SharedFrozenEncoder",
        device: str = "cuda:1",
    ) -> IQLActorBatch:
        """Sample chunks for actor-side advantage scoring (no rewards needed)."""
        start_indices = self._sample_start_indices(batch_size)
        sequences = self._gather_chunks_for_starts(start_indices)
        camera_names = list(self._base.camera_names)

        s_images: list[np.ndarray] = []
        s_proprio: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        episode_ids: list[int] = []
        episode_steps: list[int] = []
        for sequence in sequences:
            first = sequence[0]
            s_obs = first.obs
            s_images.append(_stack_views_uint8(s_obs, camera_names))
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
            context = encoder.encode(
                image_obs_raw=s_image_tensor,
                proprio_raw=s_proprio_tensor,
                action_real=action_tensor_raw[:, 0, :],
            )

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
            valid_starts = self._get_iql_valid_start_indices_locked()
        return len(valid_starts) >= int(batch_size)

    def __len__(self) -> int:
        return len(self._base)


class IQLPreencodedReplayCache:
    """Tensor-only replay cache for offline IQL warmup."""

    def __init__(
        self,
        *,
        context: torch.Tensor,
        next_context: torch.Tensor,
        action_chunk: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        is_online: torch.Tensor,
        is_intervention: torch.Tensor,
        source_size: int,
    ) -> None:
        self.context = context.contiguous()
        self.next_context = next_context.contiguous()
        self.action_chunk = action_chunk.contiguous()
        self.rewards = rewards.contiguous()
        self.dones = dones.contiguous()
        self.is_online = is_online.contiguous()
        self.is_intervention = is_intervention.contiguous()
        self.source_size = int(source_size)

        n = int(self.context.shape[0])
        for name, tensor in (
            ("next_context", self.next_context),
            ("action_chunk", self.action_chunk),
            ("rewards", self.rewards),
            ("dones", self.dones),
            ("is_online", self.is_online),
            ("is_intervention", self.is_intervention),
        ):
            if int(tensor.shape[0]) != n:
                raise ValueError(
                    f"IQLPreencodedReplayCache {name} batch dim {tensor.shape[0]} "
                    f"does not match context batch dim {n}."
                )

    def sample_step_batch(
        self,
        batch_size: int,
        *,
        encoder: "SharedFrozenEncoder | None" = None,
        discriminator: "OnlineBCEDiscriminator | None" = None,
        device: str = "cuda:1",
    ) -> IQLStepBatch:
        """Sample a cached encoded batch. Encoder/discriminator args are ignored."""
        del encoder, discriminator
        if len(self) == 0:
            raise ValueError("IQLPreencodedReplayCache is empty.")
        index_device = self.context.device
        idx = torch.randint(0, len(self), (int(batch_size),), device=index_device)
        batch = IQLStepBatch(
            context=self.context.index_select(0, idx),
            next_context=self.next_context.index_select(0, idx),
            action_chunk=self.action_chunk.index_select(0, idx),
            rewards=self.rewards.index_select(0, idx),
            dones=self.dones.index_select(0, idx),
            is_online=self.is_online.index_select(0, idx),
            is_intervention=self.is_intervention.index_select(0, idx),
            metadata={"source": "preencoded_cache"},
        )
        return batch.to(device)

    def ready(self, batch_size: int) -> bool:
        return len(self) >= int(batch_size)

    def __len__(self) -> int:
        return int(self.context.shape[0])
