from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

from .common import VASTActorBatch, VASTConfig, VASTStepBatch
from .data_util import (
    aggregate_chunk_reward,
    chunk_done_mask,
    mask_absorbing_tail_rewards,
)

# Opt-in timing instrumentation for the offline warmup preencode sweep. Enable
# with WARMUP_PROFILE=1 to print a per-stage breakdown (gather / assemble /
# encode / disc / finalize / to_cache). Off by default => zero overhead.
_WARMUP_PROFILE = os.environ.get("WARMUP_PROFILE", "0").strip().lower() not in {
    "0",
    "",
    "false",
    "no",
}
_PROFILE_TIMES: dict[str, float] = defaultdict(float)


def _sync_if_cuda(device: Any) -> None:
    if (
        isinstance(device, str)
        and device.startswith("cuda")
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize(device)

if TYPE_CHECKING:
    from robosuite.pipeline.algorithms.discriminator.encoder import SharedDynamicsEncoder
    from robosuite.pipeline.algorithms.discriminator.nnpu import FrozenNNPUDiscriminator


def _episode_index_of(transition: Any) -> int:
    info = getattr(transition, "info", None) or {}
    return int(info.get("episode_index", -1))


def _nnpu_disc_intrinsic_for_transition(transition: Any) -> float | None:
    """Read an optional precomputed nnPU intrinsic reward."""
    info = getattr(transition, "info", None) or {}
    cached = info.get("nnpu_disc_intrinsic")
    if cached is not None:
        return float(cached)
    failure_score = info.get("nnpu_failure_score")
    tau = info.get("nnpu_threshold")
    if failure_score is None or tau is None:
        return None
    return float(-torch.sigmoid(torch.tensor(float(failure_score) - float(tau))).item())


def _nnpu_disc_steps_for_sequence(sequence: list[Any], horizon: int) -> np.ndarray | None:
    steps: list[float] = []
    for item in sequence:
        intrinsic = _nnpu_disc_intrinsic_for_transition(item)
        if intrinsic is None:
            return None
        steps.append(intrinsic)
    if len(steps) != int(horizon):
        return None
    return np.asarray(steps, dtype=np.float32)


def _stack_views_uint8(obs: Any, camera_names: list[str]) -> np.ndarray:
    """Stack per-camera uint8 frames from ``obs`` into (V, 3, H, W).

    No resize or crop is applied here; the shared dynamics encoder owns its
    complete image preprocessing pipeline.
    """
    images = []
    for camera_name in camera_names:
        if camera_name not in obs:
            raise KeyError(f"VAST replay sample is missing camera '{camera_name}'")
        image = np.asarray(obs[camera_name], dtype=np.uint8)
        images.append(np.transpose(image, (2, 0, 1)))  # (3, H, W)
    return np.stack(images, axis=0)  # (V, 3, H, W)


def _to_image_tensor(stacked: np.ndarray, device: str) -> torch.Tensor:
    """(B, V, 3, H, W) uint8 -> CUDA float32 in [0, 1]."""
    if not str(device).startswith("cuda"):
        raise ValueError(f"VAST replay tensor computation requires CUDA, got {device!r}.")
    tensor = torch.from_numpy(np.ascontiguousarray(stacked)).to(
        device=device, dtype=torch.float32
    )
    return tensor.div_(255.0)


class VASTReplayBuffer:
    """Chunk-centric replay buffer wrapping a `FlowDaggerReplayBuffer`-style store.

    Args:
        base_buffer: the underlying transition store (typically the DIPOLE
            online buffer; passed by reference, not copied). Must expose
            `_storage`, `_lock`, `_get_valid_start_indices_locked()`,
            `camera_names`, `image_size`, `action_horizon`.
        cfg: VASTConfig
    """

    def __init__(self, base_buffer: Any, cfg: VASTConfig) -> None:
        self._base = base_buffer
        self.cfg = cfg
        self._vast_rng = np.random.default_rng(int(cfg.vast_sampling_seed))
        if int(getattr(base_buffer, "action_horizon", -1)) != int(cfg.action_horizon):
            raise ValueError(
                "VASTReplayBuffer: base_buffer.action_horizon "
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
                raise ValueError("VASTReplayBuffer: base buffer has no valid sequences.")
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
        encoder: "SharedDynamicsEncoder",
        discriminator: "FrozenNNPUDiscriminator | None" = None,
        device: str = "cuda:1",
    ) -> VASTStepBatch:
        """Sample `batch_size` chunk windows, encode every frame under no_grad,
        and synthesize total rewards (env + optional disc intrinsic).

        Optional precomputed nnPU rewards are read from transition metadata;
        otherwise the frozen nnPU head scores the per-frame chunk features.
        """
        return self._sample_vast_step_batch(
            batch_size,
            encoder=encoder,
            discriminator=discriminator,
            device=device,
        )

    def _raw_chunk_done_locked(self, start: int) -> bool:
        """Absorbing-terminal semantics matching ``_build_step_batch``."""
        storage = self._base._storage  # noqa: SLF001
        H = int(self.cfg.action_horizon)
        sequence = storage[start : start + H]
        first_info = sequence[0].info or {}
        if "success" in first_info:
            return any(bool((item.info or {}).get("success", False)) for item in sequence)
        if any(bool(item.done) for item in sequence):
            return True
        tail = start + H
        if tail >= len(storage):
            return True
        return _episode_index_of(storage[tail]) != _episode_index_of(storage[start])

    def _sample_vast_step_batch(
        self,
        batch_size: int,
        *,
        encoder: "SharedDynamicsEncoder",
        discriminator: "FrozenNNPUDiscriminator | None",
        device: str,
    ) -> VASTStepBatch:
        """Sample same-trajectory macro paths and encode all selected chunks."""
        H = int(self.cfg.action_horizon)
        with self._base._lock:  # noqa: SLF001
            storage = self._base._storage  # noqa: SLF001
            valid_starts = list(self._base._get_valid_start_indices_locked())  # noqa: SLF001
            if not valid_starts:
                raise ValueError("VASTReplayBuffer: base buffer has no valid sequences.")
            valid_set = set(valid_starts)
            sampled_rows = np.random.randint(0, len(valid_starts), size=int(batch_size))
            anchors = [valid_starts[int(row)] for row in sampled_rows]
            path_starts: list[list[int]] = []
            ks: list[int] = []
            js: list[int] = []
            for start in anchors:
                episode = _episode_index_of(storage[start])
                candidates: list[int] = []
                for macro_idx in range(int(self.cfg.vast_max_k)):
                    macro_start = start + macro_idx * H
                    if macro_start not in valid_set:
                        break
                    if _episode_index_of(storage[macro_start]) != episode:
                        break
                    candidates.append(macro_start)
                    if self._raw_chunk_done_locked(macro_start):
                        break
                if not candidates:
                    raise RuntimeError(f"VAST anchor {start} has no legal macro chunk.")
                k = int(self._vast_rng.integers(1, len(candidates) + 1))
                j = int(self._vast_rng.integers(1, k)) if k >= 2 else 1
                path_starts.append(candidates[:k])
                ks.append(k)
                js.append(j)
            flat_starts = [start for path in path_starts for start in path]
            sequences = [storage[start : start + H] for start in flat_starts]

        flat = self._build_step_batch(
            sequences,
            flat_starts,
            encoder=encoder,
            discriminator=discriminator,
            device=device,
        )
        current_rows: list[int] = []
        future_rows: list[int] = []
        intermediate_rows: list[int] = []
        returns: list[torch.Tensor] = []
        path_dones: list[torch.Tensor] = []
        offset = 0
        gamma_h = float(self.cfg.discount) ** H
        for k, j in zip(ks, js):
            rows = list(range(offset, offset + k))
            current_rows.append(rows[0])
            future_rows.append(rows[-1])
            intermediate_rows.append(rows[j] if k >= 2 else rows[0])
            discounts = torch.tensor(
                [gamma_h**m for m in range(k)],
                device=flat.rewards.device,
                dtype=flat.rewards.dtype,
            ).reshape(-1, 1)
            returns.append((flat.rewards[rows] * discounts).sum(dim=0))
            path_dones.append(flat.dones[rows].amax(dim=0))
            offset += k

        current_idx = torch.tensor(current_rows, device=flat.rewards.device, dtype=torch.long)
        future_idx = torch.tensor(future_rows, device=flat.rewards.device, dtype=torch.long)
        intermediate_idx = torch.tensor(
            intermediate_rows, device=flat.rewards.device, dtype=torch.long
        )
        k_tensor = torch.tensor(ks, device=flat.rewards.device, dtype=flat.rewards.dtype).view(-1, 1)
        j_tensor = torch.tensor(js, device=flat.rewards.device, dtype=flat.rewards.dtype).view(-1, 1)
        return VASTStepBatch(
            chunk_feature=flat.chunk_feature.index_select(0, current_idx),
            v_state_feature=flat.v_state_feature.index_select(0, current_idx),
            next_v_state_feature=flat.next_v_state_feature.index_select(0, current_idx),
            action_chunk=flat.action_chunk.index_select(0, current_idx),
            rewards=flat.rewards.index_select(0, current_idx),
            dones=flat.dones.index_select(0, current_idx),
            is_online=flat.is_online.index_select(0, current_idx),
            is_intervention=flat.is_intervention.index_select(0, current_idx),
            metadata={
                "start_indices": anchors,
                "future_start_indices": [path[-1] + H for path in path_starts],
                "sampled_k": ks,
                "sampled_j": js,
            },
            future_v_state_feature=flat.next_v_state_feature.index_select(0, future_idx),
            intermediate_v_state_feature=flat.v_state_feature.index_select(
                0, intermediate_idx
            ),
            k=k_tensor,
            j=j_tensor,
            k_step_returns=torch.stack(returns, dim=0),
            mc_mask=(k_tensor >= 2).to(flat.rewards.dtype),
            future_dones=torch.stack(path_dones, dim=0),
        ).to(device)

    def _build_step_batch(
        self,
        sequences: list[list[Any]],
        start_indices: list[int],
        *,
        encoder: "SharedDynamicsEncoder",
        discriminator: "FrozenNNPUDiscriminator | None" = None,
        device: str = "cuda:1",
    ) -> VASTStepBatch:
        """Encode the given chunk ``sequences`` (with their ``start_indices``)
        and build an :class:`VASTStepBatch`.

        Shared by :meth:`sample_step_batch` (random draw) and
        :meth:`preencode_step_cache` (deterministic full sweep). Under the
        frozen encoder this is a pure, deterministic function of its inputs, so
        the resulting batch is independent of *how* the start indices were
        chosen — that is what makes the preencode cache results-neutral.
        """
        _prof = _WARMUP_PROFILE
        _t0 = time.perf_counter() if _prof else 0.0

        camera_names = list(self._base.camera_names)
        H = int(self.cfg.action_horizon)

        chunk_images: list[np.ndarray] = []     # each (H, V, 3, h, w)
        chunk_proprio: list[np.ndarray] = []    # each (H, D_s)
        sp_images: list[np.ndarray] = []        # each (V, 3, h, w)
        sp_proprio: list[np.ndarray] = []       # each (D_s,)
        actions: list[np.ndarray] = []          # each (H, D_a)
        rewards_per_step: list[np.ndarray] = []
        dones_per_step: list[np.ndarray] = []
        success_per_step: list[np.ndarray] = []
        post_success_per_step: list[np.ndarray] = []
        is_online: list[float] = []
        is_intervention: list[float] = []
        episode_ids: list[int] = []
        episode_steps: list[int] = []
        nnpu_disc_per_sequence: list[np.ndarray | None] = []

        for sequence, start in zip(sequences, start_indices):
            nnpu_disc_per_sequence.append(_nnpu_disc_steps_for_sequence(sequence, H))
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
            # Bootstrap terminal mask: γ^H·V(s') is dropped (done=1) ONLY at a
            # genuine absorbing terminal. For offline demos that means a task
            # *success* frame — a trajectory that merely ran off its recording
            # boundary (e.g. every fail_rollout) is a *truncation*, so we keep
            # bootstrapping and its tail chunks regress to r + γ^H·V(s') like the
            # interior instead of collapsing to the immediate chunk reward.
            # `_next_obs_for` already supplies a valid s', so `forced_done` must
            # not zero the bootstrap on the success-annotated (offline) path.
            if "success" in (first.info or {}):
                step_dones = np.asarray(
                    [bool((item.info or {}).get("success", False)) for item in sequence],
                    dtype=np.float32,
                )
            else:
                # Online / unknown provenance: preserve the prior done semantics.
                step_dones = np.asarray([bool(item.done) for item in sequence], dtype=np.float32)
                if forced_done:
                    step_dones[-1] = 1.0
            rewards_per_step.append(step_rewards)
            dones_per_step.append(step_dones)
            success_per_step.append(
                np.asarray(
                    [bool((item.info or {}).get("success", False)) for item in sequence],
                    dtype=np.float32,
                )
            )
            post_success_per_step.append(
                np.asarray(
                    [
                        bool((item.info or {}).get("frozen_post_success", False))
                        or bool(
                            (item.info or {}).get("synthetic_vast_success_tail", False)
                        )
                        for item in sequence
                    ],
                    dtype=np.float32,
                )
            )
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
        success_np = np.stack(success_per_step, axis=0)             # (B, H)
        post_success_np = np.stack(post_success_per_step, axis=0)   # (B, H)

        V, C, Hi, Wi = chunk_images_np.shape[2:]
        chunk_images_tensor = _to_image_tensor(
            np.ascontiguousarray(chunk_images_np.reshape(B * H, V, C, Hi, Wi)),
            device,
        ).view(B, H, V, C, Hi, Wi)
        chunk_proprio_tensor = torch.from_numpy(np.ascontiguousarray(chunk_proprio_np)).to(
            device=device, dtype=torch.float32
        )
        action_tensor = torch.from_numpy(np.ascontiguousarray(action_np)).to(
            device=device, dtype=torch.float32
        )
        reward_tensor = torch.from_numpy(np.ascontiguousarray(rewards_np)).to(
            device=device, dtype=torch.float32
        )
        done_tensor = torch.from_numpy(np.ascontiguousarray(dones_np)).to(
            device=device, dtype=torch.float32
        )
        success_tensor = torch.from_numpy(np.ascontiguousarray(success_np)).to(
            device=device, dtype=torch.float32
        )
        post_success_tensor = torch.from_numpy(
            np.ascontiguousarray(post_success_np)
        ).to(device=device, dtype=torch.float32)
        is_online_tensor = torch.tensor(
            is_online, device=device, dtype=torch.float32
        ).unsqueeze(-1)
        is_intervention_tensor = torch.tensor(
            is_intervention, device=device, dtype=torch.float32
        ).unsqueeze(-1)

        sp_image_tensor = _to_image_tensor(sp_images_np, device)                # (B, V, 3, Hi, Wi)
        sp_proprio_tensor = torch.from_numpy(np.ascontiguousarray(sp_proprio_np)).to(
            device=device, dtype=torch.float32
        )
        if _prof:
            _PROFILE_TIMES["assemble"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()

        # Only the *first* chunk frame's VAST features (plus the s' state feature)
        # feed the VAST batch (`chunk_feature`/`v_state_feature` index [:, 0, :]).
        # The frozen encoder's per-frame disc reward is the sole consumer of the
        # remaining H-1 chunk frames, so when no live disc scoring is needed we
        # encode just frame 0 of the chunk instead of all H. Each frame is encoded
        # independently along the batch dim, so the frame-0 output is bit-identical
        # either way — this is a pure speedup (~H x fewer DINOv3 forwards on the
        # chunk path; the dominant warmup cost).
        effective_disc_coef = float(self.cfg.disc_reward_coef)
        effective_output_coef = float(self.cfg.output_reward_coef)
        use_precomputed_disc = (
            effective_disc_coef != 0.0
            and len(nnpu_disc_per_sequence) == B
            and all(step is not None for step in nnpu_disc_per_sequence)
        )
        need_full_chunk_features = (
            effective_disc_coef != 0.0
            and not use_precomputed_disc
            and discriminator is not None
        )

        chunk_features: torch.Tensor | None = None
        with torch.no_grad():
            if need_full_chunk_features:
                state_features, chunk_features = encoder.encode_features(
                    chunk_images=chunk_images_tensor,
                    chunk_proprio=chunk_proprio_tensor,
                    chunk_actions=action_tensor,
                )
                chunk_feature = chunk_features[:, 0, :]
                v_state_feature = state_features[:, 0, :]
            else:
                # frame-0-only fast path: encode obs_0 once, fused with the full
                # H-step action window (identical to encode_features step 0).
                v_state_feature, chunk_feature = encoder.encode_state_and_chunk(
                    image_obs_raw=chunk_images_tensor[:, 0],
                    proprio_raw=chunk_proprio_tensor[:, 0],
                    action_chunk=action_tensor,
                )
            next_v_state_feature = encoder.encode_state(
                image_obs_raw=sp_image_tensor,
                proprio_raw=sp_proprio_tensor,
            )
        if _prof:
            _sync_if_cuda(device)
            _PROFILE_TIMES["encode"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()

        # Per-frame disc reward: optional cache or frozen nnPU head.
        if effective_disc_coef == 0.0:
            r_disc_per_step = torch.zeros_like(reward_tensor)
        elif use_precomputed_disc:
            r_disc_per_step = torch.from_numpy(
                np.stack([step for step in nnpu_disc_per_sequence if step is not None])
            ).to(device=reward_tensor.device, dtype=reward_tensor.dtype)
        elif discriminator is not None:
            with torch.no_grad():
                r_disc_per_step = discriminator.intrinsic_reward(
                    chunk_feature=chunk_features
                )
            r_disc_per_step = r_disc_per_step.view(B, H).to(
                reward_tensor.device, dtype=reward_tensor.dtype
            )
        else:
            r_disc_per_step = torch.zeros_like(reward_tensor)

        if _prof:
            _sync_if_cuda(device)
            _PROFILE_TIMES["disc"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()

        r_total_chunk = effective_output_coef * reward_tensor + effective_disc_coef * r_disc_per_step
        r_total_chunk = mask_absorbing_tail_rewards(
            r_total_chunk,
            success_tensor,
            post_success_tensor,
        )
        rewards = aggregate_chunk_reward(r_total_chunk, float(self.cfg.discount))
        dones = chunk_done_mask(done_tensor)

        disc_meta: dict[str, float] = {}
        if effective_disc_coef != 0.0 and (use_precomputed_disc or discriminator is not None):
            effective_disc_reward = mask_absorbing_tail_rewards(
                r_disc_per_step,
                success_tensor,
                post_success_tensor,
            )
            disc_meta["disc_reward_first_frame_mean"] = float(
                effective_disc_reward[:, 0].mean().item()
            )
            disc_meta["disc_reward_chunk_mean"] = float(
                effective_disc_reward.mean().item()
            )

        batch = VASTStepBatch(
            chunk_feature=chunk_feature,
            v_state_feature=v_state_feature,
            next_v_state_feature=next_v_state_feature,
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
        out = batch.to(device)
        if _prof:
            _sync_if_cuda(device)
            _PROFILE_TIMES["finalize"] += time.perf_counter() - _t0
        return out

    def sample_actor_batch(
        self,
        batch_size: int,
        *,
        encoder: "SharedDynamicsEncoder",
        device: str = "cuda:1",
    ) -> VASTActorBatch:
        """Sample chunks for actor-side advantage scoring (no rewards needed)."""
        sequences, start_indices = self._gather_chunks(batch_size)
        camera_names = list(self._base.camera_names)

        chunk_images: list[np.ndarray] = []
        chunk_proprio: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        episode_ids: list[int] = []
        episode_steps: list[int] = []
        for sequence in sequences:
            first = sequence[0]
            chunk_images.append(
                np.stack([_stack_views_uint8(item.obs, camera_names) for item in sequence])
            )
            chunk_proprio.append(
                np.stack([np.asarray(item.obs["state"], dtype=np.float32) for item in sequence])
            )
            actions.append(
                np.stack([np.asarray(item.action, dtype=np.float32) for item in sequence], axis=0)
            )
            info = first.info or {}
            episode_ids.append(int(info.get("episode_index", -1)))
            episode_steps.append(int(info.get("episode_step", -1)))

        image_np = np.stack(chunk_images, axis=0)
        B, H, V, C, Hi, Wi = image_np.shape
        image_tensor = _to_image_tensor(
            np.ascontiguousarray(image_np.reshape(B * H, V, C, Hi, Wi)), device
        ).view(B, H, V, C, Hi, Wi)
        proprio_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(chunk_proprio, axis=0))
        ).to(device=device, dtype=torch.float32)
        action_tensor_raw = torch.from_numpy(
            np.ascontiguousarray(np.stack(actions, axis=0))
        ).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            state_features, chunk_features = encoder.encode_features(
                chunk_images=image_tensor,
                chunk_proprio=proprio_tensor,
                chunk_actions=action_tensor_raw,
            )

        batch = VASTActorBatch(
            chunk_feature=chunk_features[:, 0, :],
            v_state_feature=state_features[:, 0, :],
            action_chunk=action_tensor_raw,
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

    # ------------------------------------------------------------------ #
    # Pre-encoded cache (warmup acceleration; results-neutral)            #
    # ------------------------------------------------------------------ #

    def preencode_step_cache(
        self,
        *,
        encoder: "SharedDynamicsEncoder",
        discriminator: "FrozenNNPUDiscriminator | None" = None,
        device: str = "cuda:1",
        encode_batch_size: int = 64,
        cache_device: str = "cpu",
        progress_desc: str | None = None,
    ) -> "VASTPreencodedReplayCache":
        """Encode *every* valid chunk once and return an in-memory cache.

        This is a pure training-speed optimization for the offline warmup
        setting (frozen encoder, static dataset, static nnPU rewards). The
        frozen encoder runs under ``no_grad`` and is deterministic, so the
        cached tensors are bit-for-bit what :meth:`sample_step_batch` would
        produce for the same start index. The cache is built in
        ``_get_valid_start_indices_locked()`` order, so cache row ``i``
        corresponds to ``valid_starts[i]`` — letting
        :class:`VASTPreencodedReplayCache` reproduce the same uniform sampling
        (and hence the same RNG stream and the same training result).
        """
        with self._base._lock:  # noqa: SLF001
            valid_starts = list(self._base._get_valid_start_indices_locked())  # noqa: SLF001
        if not valid_starts:
            raise ValueError(
                "VASTReplayBuffer: base buffer has no valid sequences to preencode."
            )
        H = int(self.cfg.action_horizon)
        encode_bs = max(1, int(encode_batch_size))

        field_names = (
            "chunk_feature",
            "v_state_feature",
            "next_v_state_feature",
            "action_chunk",
            "rewards",
            "dones",
            "is_online",
            "is_intervention",
        )
        parts: dict[str, list[torch.Tensor]] = {name: [] for name in field_names}

        iterator: Any = range(0, len(valid_starts), encode_bs)
        if progress_desc is not None:
            iterator = tqdm(
                iterator,
                total=(len(valid_starts) + encode_bs - 1) // encode_bs,
                desc=progress_desc,
            )
        if _WARMUP_PROFILE:
            _PROFILE_TIMES.clear()
        for offset in iterator:
            _t = time.perf_counter() if _WARMUP_PROFILE else 0.0
            batch_starts = valid_starts[offset : offset + encode_bs]
            with self._base._lock:  # noqa: SLF001
                sequences = [self._base._storage[s : s + H] for s in batch_starts]  # noqa: SLF001
            if _WARMUP_PROFILE:
                _PROFILE_TIMES["gather"] += time.perf_counter() - _t
            batch = self._build_step_batch(
                sequences,
                batch_starts,
                encoder=encoder,
                discriminator=discriminator,
                device=device,
            )
            _t = time.perf_counter() if _WARMUP_PROFILE else 0.0
            # Assemble the cache on CPU even when its final home is CUDA.  The
            # old CUDA path retained every per-batch tensor and then allocated
            # a second full-sized torch.cat output, briefly doubling the cache
            # VRAM footprint.  A CPU assembly followed by one exact device copy
            # keeps peak VRAM close to the final cache size.
            for name in field_names:
                parts[name].append(getattr(batch, name).detach().to("cpu"))
            if _WARMUP_PROFILE:
                _PROFILE_TIMES["to_cache"] += time.perf_counter() - _t

        if _WARMUP_PROFILE:
            total = sum(_PROFILE_TIMES.values()) or 1e-9
            print("[warmup][profile] preencode stage breakdown:")
            for key, value in sorted(_PROFILE_TIMES.items(), key=lambda kv: -kv[1]):
                print(f"  {key:10s} {value:8.2f}s ({100.0 * value / total:5.1f}%)")
            print(f"  {'TOTAL':10s} {total:8.2f}s")

        assembled = {
            name: torch.cat(parts[name], dim=0) for name in field_names
        }
        target_cache_device = torch.device(cache_device)
        if target_cache_device.type != "cpu":
            assembled = {
                name: tensor.to(target_cache_device)
                for name, tensor in assembled.items()
            }

        return VASTPreencodedReplayCache(
            chunk_feature=assembled["chunk_feature"],
            v_state_feature=assembled["v_state_feature"],
            next_v_state_feature=assembled["next_v_state_feature"],
            action_chunk=assembled["action_chunk"],
            rewards=assembled["rewards"],
            dones=assembled["dones"],
            is_online=assembled["is_online"],
            is_intervention=assembled["is_intervention"],
            source_size=len(valid_starts),
            cfg=self.cfg,
            valid_starts=valid_starts,
            episode_ids=[
                _episode_index_of(self._base._storage[start])  # noqa: SLF001
                for start in valid_starts
            ],
        )


class VASTPreencodedReplayCache:
    """In-memory cache of pre-encoded :class:`VASTStepBatch` tensors.

    Built by :meth:`VASTReplayBuffer.preencode_step_cache`. Exposes a
    ``sample_step_batch`` interface compatible with :class:`VASTReplayBuffer`
    (it accepts and ignores ``encoder`` / ``discriminator`` so call sites need
    no change), but samples by indexing into the cached tensors instead of
    re-running the frozen encoder every step.

    Sampling uses the same ``np.random.randint(0, N, batch_size)`` draw as
    :meth:`VASTReplayBuffer._gather_chunks`, so against an identical RNG state it
    selects the same chunks — making cached warmup byte-for-byte equivalent to
    the live path.
    """

    def __init__(
        self,
        *,
        chunk_feature: torch.Tensor,
        v_state_feature: torch.Tensor,
        next_v_state_feature: torch.Tensor,
        action_chunk: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        is_online: torch.Tensor,
        is_intervention: torch.Tensor,
        source_size: int,
        cfg: VASTConfig,
        valid_starts: list[int],
        episode_ids: list[int],
    ) -> None:
        self.cfg = cfg
        self._vast_rng = np.random.default_rng(int(cfg.vast_sampling_seed))
        self.chunk_feature = chunk_feature.contiguous()
        self.v_state_feature = v_state_feature.contiguous()
        self.next_v_state_feature = next_v_state_feature.contiguous()
        self.action_chunk = action_chunk.contiguous()
        self.rewards = rewards.contiguous()
        self.dones = dones.contiguous()
        self.is_online = is_online.contiguous()
        self.is_intervention = is_intervention.contiguous()
        self.source_size = int(source_size)
        self.valid_starts = tuple(int(v) for v in valid_starts)
        self.episode_ids = tuple(int(v) for v in episode_ids)
        self._start_to_row = {start: row for row, start in enumerate(self.valid_starts)}

        # VAST path legality depends only on the static cache metadata.  Build
        # the row table once instead of repeating dict lookups, episode checks,
        # and scalar Tensor.item() calls for every sample in every update.
        # Keeping k/j draws in the sampling loop below preserves the exact
        # Generator call order (and therefore the sampled sequence) unchanged.
        self._vast_path_rows: np.ndarray | None = None
        self._vast_path_lengths: np.ndarray | None = None
        self._build_vast_path_table()

        n = self.chunk_feature.shape[0]
        for name, tensor in (
            ("v_state_feature", self.v_state_feature),
            ("next_v_state_feature", self.next_v_state_feature),
            ("action_chunk", self.action_chunk),
            ("rewards", self.rewards),
            ("dones", self.dones),
            ("is_online", self.is_online),
            ("is_intervention", self.is_intervention),
        ):
            if tensor.shape[0] != n:
                raise ValueError(
                    f"VASTPreencodedReplayCache {name} batch dim {tensor.shape[0]} "
                    f"!= chunk_feature batch dim {n}."
                )

    def _build_vast_path_table(self) -> None:
        """Precompute legal same-episode macro paths for every cache row."""
        H = int(self.cfg.action_horizon)
        max_k = int(self.cfg.vast_max_k)
        n = len(self.valid_starts)
        path_rows = np.full((n, max_k), -1, dtype=np.int64)
        path_lengths = np.zeros(n, dtype=np.int64)
        done_rows = (
            self.dones.detach().amax(dim=1).to(device="cpu").numpy().astype(bool, copy=False)
        )
        for row, (start, episode) in enumerate(
            zip(self.valid_starts, self.episode_ids)
        ):
            for macro_idx in range(max_k):
                candidate = self._start_to_row.get(start + macro_idx * H)
                if candidate is None or self.episode_ids[candidate] != episode:
                    break
                path_rows[row, macro_idx] = candidate
                path_lengths[row] += 1
                if done_rows[candidate]:
                    break
            if path_lengths[row] == 0:
                raise RuntimeError(f"VAST cache row {row} has no legal macro chunk.")
        self._vast_path_rows = path_rows
        self._vast_path_lengths = path_lengths

    def sample_step_batch(
        self,
        batch_size: int,
        *,
        encoder: "SharedDynamicsEncoder | None" = None,
        discriminator: "FrozenNNPUDiscriminator | None" = None,
        device: str = "cuda:1",
    ) -> VASTStepBatch:
        # `encoder` / `discriminator` are accepted for call-site compatibility
        # and intentionally unused (everything is already encoded).
        del encoder, discriminator
        n = self.chunk_feature.shape[0]
        if n == 0:
            raise ValueError("VASTPreencodedReplayCache is empty.")
        sampled = np.random.randint(0, n, size=int(batch_size))
        return self._sample_vast_step_batch(sampled, device=device)

    def _sample_vast_step_batch(
        self, sampled_rows: np.ndarray, *, device: str
    ) -> VASTStepBatch:
        H = int(self.cfg.action_horizon)
        if self._vast_path_rows is None or self._vast_path_lengths is None:
            raise RuntimeError("VAST path table was not initialized.")
        sampled_rows = np.asarray(sampled_rows, dtype=np.int64)
        path_lengths = self._vast_path_lengths[sampled_rows]
        ks = np.empty(sampled_rows.shape[0], dtype=np.int64)
        js = np.empty(sampled_rows.shape[0], dtype=np.int64)
        # Do not vectorize these RNG calls: the conditional j draw is part of
        # the established reproducibility contract.
        for index, path_length in enumerate(path_lengths):
            k = int(self._vast_rng.integers(1, int(path_length) + 1))
            j = int(self._vast_rng.integers(1, k)) if k >= 2 else 1
            ks[index] = k
            js[index] = j

        all_path_rows = self._vast_path_rows[sampled_rows]
        batch_rows = np.arange(sampled_rows.shape[0], dtype=np.int64)
        last_rows = all_path_rows[batch_rows, ks - 1]
        middle_offsets = np.where(ks >= 2, js, 0)
        middle_rows = all_path_rows[batch_rows, middle_offsets]

        cache_device = self.chunk_feature.device
        current = torch.from_numpy(sampled_rows).to(device=cache_device)
        last = torch.from_numpy(last_rows).to(device=cache_device)
        middle = torch.from_numpy(middle_rows).to(device=cache_device)
        path_index = torch.from_numpy(all_path_rows).to(device=cache_device)
        safe_path_index = path_index.clamp_min(0)
        flat_path_index = safe_path_index.reshape(-1)
        path_rewards = self.rewards.index_select(0, flat_path_index).view(
            sampled_rows.shape[0], int(self.cfg.vast_max_k), -1
        )
        path_dones = self.dones.index_select(0, flat_path_index).view(
            sampled_rows.shape[0], int(self.cfg.vast_max_k), -1
        )
        gamma_h = float(self.cfg.discount) ** H
        returns = torch.zeros_like(path_rewards[:, 0])
        # Preserve the legacy left-to-right floating-point accumulation order,
        # but perform each horizon position for the whole batch at once.
        for macro_idx in range(int(self.cfg.vast_max_k)):
            active = (ks > macro_idx)
            if not bool(active.any()):
                break
            active_tensor = torch.from_numpy(active).to(
                device=cache_device, dtype=self.rewards.dtype
            ).view(-1, 1)
            returns = returns + (
                path_rewards[:, macro_idx] * (gamma_h**macro_idx) * active_tensor
            )
        sampled_path_mask = torch.arange(
            int(self.cfg.vast_max_k), device=cache_device
        ).view(1, -1) < torch.from_numpy(ks).to(device=cache_device).view(-1, 1)
        future_dones = path_dones.masked_fill(
            ~sampled_path_mask.unsqueeze(-1), 0
        ).amax(dim=1)
        k_tensor = torch.from_numpy(ks).to(
            device=cache_device, dtype=self.rewards.dtype
        ).view(-1, 1)
        j_tensor = torch.from_numpy(js).to(
            device=cache_device, dtype=self.rewards.dtype
        ).view(-1, 1)
        batch = VASTStepBatch(
            chunk_feature=self.chunk_feature.index_select(0, current),
            v_state_feature=self.v_state_feature.index_select(0, current),
            next_v_state_feature=self.next_v_state_feature.index_select(0, current),
            action_chunk=self.action_chunk.index_select(0, current),
            rewards=self.rewards.index_select(0, current),
            dones=self.dones.index_select(0, current),
            is_online=self.is_online.index_select(0, current),
            is_intervention=self.is_intervention.index_select(0, current),
            metadata={
                "source": "preencoded_cache",
                "start_indices": [self.valid_starts[int(row)] for row in sampled_rows],
                "future_start_indices": [self.valid_starts[int(row)] + H for row in last_rows],
                "sampled_k": ks.tolist(),
                "sampled_j": js.tolist(),
            },
            future_v_state_feature=self.next_v_state_feature.index_select(0, last),
            intermediate_v_state_feature=self.v_state_feature.index_select(0, middle),
            k=k_tensor,
            j=j_tensor,
            k_step_returns=returns,
            mc_mask=(k_tensor >= 2).to(self.rewards.dtype),
            future_dones=future_dones,
        )
        return batch.to(device)

    def ready(self, batch_size: int) -> bool:
        return self.chunk_feature.shape[0] >= int(batch_size)

    def __len__(self) -> int:
        return int(self.chunk_feature.shape[0])
