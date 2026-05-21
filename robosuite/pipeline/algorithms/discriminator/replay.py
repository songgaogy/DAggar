"""Balanced replay buffer for online discriminator training.

Wraps the trainer's base online buffer (any `FlowDaggerReplayBuffer`-style
store) by reference and partitions valid chunk-start indices into two
buckets:

    failure_starts     : chunks where any step in the H-length window has
                         `is_intervention=True`. Human intervention is the
                         positive (label=1) class — the policy failed and
                         the human had to take over.
    non_failure_starts : everything else (offline expert demos, successful
                         on-policy rollouts, ...). Label=0.

Sign convention matches the warm-started lpb_v2 BCE head: higher logit =
more failure-like / more intervention-like.

This mirrors `IQLReplayBuffer`'s windowing pattern (see
`robosuite/pipeline/algorithms/q_learning/replay.py`) so there is a
single source of truth for the underlying frames.

The buffer never stores encoder latents: it runs the frozen
`SharedFrozenEncoder` at sample time so the encoder / camera binding can
change without invalidating the buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from robosuite.pipeline.algorithms.q_learning.replay import (
    _stack_views_uint8,
    _to_image_tensor,
)

from .base import DiscriminatorBatch
from .online_bce import DiscriminatorConfig

if TYPE_CHECKING:
    from robosuite.pipeline.common.types import Transition

    from .encoder import SharedFrozenEncoder


class DiscriminatorReplayBuffer:
    """Balanced two-bucket sampler over the shared online transition store.

    Args:
        cfg:            DiscriminatorConfig (uses batch_size, balance_ratio).
        base_buffer:    underlying store. Must expose `_storage`, `_lock`,
                        `_get_valid_start_indices_locked()`, `camera_names`,
                        `image_size`, `action_horizon`, `add(transition)`.
        encoder:        SharedFrozenEncoder (`bind_policy_cameras` must
                        already have been called by the trainer).
        action_horizon: H — must equal `base_buffer.action_horizon`.

    Label convention:
        label = 1.0 → intervention chunk → policy was failing
        label = 0.0 → demo or non-intervention on-policy chunk → good behavior
    """

    def __init__(
        self,
        cfg: DiscriminatorConfig,
        base_buffer: Any,
        *,
        encoder: "SharedFrozenEncoder",
        action_horizon: int,
    ) -> None:
        self.cfg = cfg
        self._base = base_buffer
        self._encoder = encoder
        self.action_horizon = int(action_horizon)
        base_H = int(getattr(base_buffer, "action_horizon", -1))
        if base_H != self.action_horizon:
            raise ValueError(
                "DiscriminatorReplayBuffer: base_buffer.action_horizon "
                f"({base_H}) must equal action_horizon ({self.action_horizon})."
            )

        self._failure_starts: list[int] = []
        self._non_failure_starts: list[int] = []
        # Track which valid_starts have already been classified so refresh
        # is incremental. We key on (storage_len, valid_starts_tuple)
        # — robust to ring-buffer rebuild because the cache invalidates on
        # length change.
        self._last_valid_starts_signature: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------ #
    # Bookkeeping                                                         #
    # ------------------------------------------------------------------ #

    def _expected_balance(self) -> tuple[int, int]:
        bs = int(self.cfg.batch_size)
        return self._split_counts(bs)

    def _split_counts(self, batch_size: int) -> tuple[int, int]:
        """Returns (n_failure, n_non_failure). `balance_ratio` is the
        failure : non_failure ratio (default 1.0 → 50/50).
        """
        ratio = float(self.cfg.balance_ratio)
        n_failure = int(round(batch_size * ratio / (1.0 + ratio)))
        n_failure = max(0, min(batch_size, n_failure))
        n_non_failure = batch_size - n_failure
        return n_failure, n_non_failure

    def _refresh_buckets(self) -> None:
        with self._base._lock:  # noqa: SLF001
            valid_starts = list(self._base._get_valid_start_indices_locked())  # noqa: SLF001
            storage = list(self._base._storage)  # noqa: SLF001
        sig = (len(storage), len(valid_starts))
        if sig == self._last_valid_starts_signature and (
            self._failure_starts or self._non_failure_starts
        ):
            return
        H = self.action_horizon
        failure: list[int] = []
        non_failure: list[int] = []
        for s in valid_starts:
            window = storage[s : s + H]
            if len(window) < H:
                continue
            is_failure = any(bool(item.is_intervention) for item in window)
            if is_failure:
                failure.append(int(s))
            else:
                non_failure.append(int(s))
        self._failure_starts = failure
        self._non_failure_starts = non_failure
        self._last_valid_starts_signature = sig

    # ------------------------------------------------------------------ #
    # Mutation                                                            #
    # ------------------------------------------------------------------ #

    def add_from_transition(self, t: "Transition") -> None:
        """No-op: classification is deferred to `_refresh_buckets()` at
        sample time. The base buffer is the single source of truth and is
        populated by `DipoleTrainer.record_transition`.
        """
        return

    def bootstrap_from_demos(self, demos: list["Transition"]) -> None:
        """Append expert demos into the underlying base buffer.

        Demos carry `is_intervention=False` (they are reference behavior,
        not corrections) and therefore fall into the *non-failure* pool
        (label=0). This matches the lpb_v2 BCE-head warm-start convention
        where higher logit = more failure-like; demos contribute the
        non-failure half of the BCE objective.
        """
        if not demos:
            return
        for t in demos:
            self._base.add(t)
        # Force a recompute so the freshly added demos are reflected in
        # non_failure_starts on next sample.
        self._last_valid_starts_signature = (-1, -1)

    # ------------------------------------------------------------------ #
    # Sampling                                                            #
    # ------------------------------------------------------------------ #

    def _sample_indices(self, pool: list[int], n: int, *, allow_replacement: bool) -> list[int]:
        if n <= 0 or not pool:
            return []
        if allow_replacement or len(pool) < n:
            idx = np.random.randint(0, len(pool), size=int(n))
        else:
            idx = np.random.choice(len(pool), size=int(n), replace=False)
        return [pool[int(i)] for i in idx]

    def sample(
        self,
        batch_size: int,
        *,
        device: str = "cuda:1",
    ) -> DiscriminatorBatch:
        self._refresh_buckets()
        n_failure, n_non_failure = self._split_counts(batch_size)
        if n_failure > 0 and not self._failure_starts:
            raise RuntimeError(
                "DiscriminatorReplayBuffer.sample: failure pool is empty. "
                "Wait until at least one intervention chunk is recorded "
                "before calling sample (gate with `ready(batch_size)`)."
            )
        if n_non_failure > 0 and not self._non_failure_starts:
            raise RuntimeError(
                "DiscriminatorReplayBuffer.sample: non_failure pool is empty. "
                "Call bootstrap_from_demos(...) or wait for a non-intervention "
                "rollout."
            )

        failure_starts = self._sample_indices(
            self._failure_starts, n_failure, allow_replacement=True
        )
        non_failure_starts = self._sample_indices(
            self._non_failure_starts,
            n_non_failure,
            allow_replacement=len(self._non_failure_starts) < n_non_failure,
        )
        all_starts = failure_starts + non_failure_starts

        H = self.action_horizon
        camera_names = list(self._base.camera_names)
        image_size = int(self._base.image_size)

        s_images: list[np.ndarray] = []
        s_proprio: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        with self._base._lock:  # noqa: SLF001
            storage = self._base._storage  # noqa: SLF001
            for start in all_starts:
                sequence = storage[start : start + H]
                first = sequence[0]
                s_obs = first.obs
                s_images.append(_stack_views_uint8(s_obs, camera_names, image_size))
                s_proprio.append(np.asarray(s_obs["state"], dtype=np.float32))
                actions.append(
                    np.stack(
                        [np.asarray(item.action, dtype=np.float32) for item in sequence],
                        axis=0,
                    )
                )

        s_image_tensor = _to_image_tensor(np.stack(s_images, axis=0), device)
        s_proprio_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(s_proprio, axis=0))
        ).float()
        action_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(actions, axis=0))
        ).float()

        with torch.no_grad():
            context = self._encoder.encode(
                image_obs_raw=s_image_tensor, proprio_raw=s_proprio_tensor
            )

        labels = torch.zeros(len(all_starts), dtype=torch.float32)
        # First `len(failure_starts)` rows correspond to intervention
        # chunks → label=1 (failure / positive class).
        labels[: len(failure_starts)] = 1.0

        target_device = torch.device(device)
        return DiscriminatorBatch(
            context=context.to(target_device),
            action_chunk=action_tensor.to(target_device),
            label=labels.to(target_device),
        )

    def ready(self, batch_size: int) -> bool:
        self._refresh_buckets()
        n_failure, n_non_failure = self._split_counts(batch_size)
        if n_failure > 0 and not self._failure_starts:
            return False
        if n_non_failure > 0 and not self._non_failure_starts:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        self._refresh_buckets()
        return len(self._failure_starts) + len(self._non_failure_starts)

    @property
    def num_failure(self) -> int:
        self._refresh_buckets()
        return len(self._failure_starts)

    @property
    def num_non_failure(self) -> int:
        self._refresh_buckets()
        return len(self._non_failure_starts)
