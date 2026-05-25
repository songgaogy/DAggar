"""Balanced per-frame replay buffer for online discriminator training.

Wraps the trainer's base online buffer (any `FlowDaggerReplayBuffer`-style
store) by reference. With the single-frame `OnlineBCEDiscriminator`, the
buffer indexes individual `Transition`s — not H-step chunk windows — and
partitions them into:

    failure_starts     : transitions where `is_intervention=True`. The
                         policy was failing on this frame and the human
                         had to take over. Label = 1 (positive class).
    non_failure_starts : everything else (offline expert demos, successful
                         on-policy rollouts, ...). Label = 0.

Sign convention matches the warm-started lpb_v2 BCE head: higher logit =
more failure-like / more intervention-like.

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
    """Balanced per-frame sampler over the shared online transition store.

    Args:
        cfg:            DiscriminatorConfig (uses batch_size, balance_ratio).
        base_buffer:    underlying store. Must expose `_storage`, `_lock`,
                        `camera_names`, `add(transition)`.
        encoder:        SharedFrozenEncoder (`bind_policy_cameras` must
                        already have been called by the trainer).

    Label convention:
        label = 1.0 → intervention frame → policy was failing
        label = 0.0 → demo or non-intervention on-policy frame → good behavior
    """

    def __init__(
        self,
        cfg: DiscriminatorConfig,
        base_buffer: Any,
        *,
        encoder: "SharedFrozenEncoder",
    ) -> None:
        self.cfg = cfg
        self._base = base_buffer
        self._encoder = encoder

        self._failure_starts: list[int] = []
        self._non_failure_starts: list[int] = []
        # Cached classification keyed on storage length; cache invalidates
        # whenever the underlying buffer grows or rebuilds.
        self._last_storage_len: int = -1

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
            storage = list(self._base._storage)  # noqa: SLF001
        if (
            len(storage) == self._last_storage_len
            and (self._failure_starts or self._non_failure_starts)
        ):
            return
        failure: list[int] = []
        non_failure: list[int] = []
        for idx, item in enumerate(storage):
            if bool(getattr(item, "is_intervention", False)):
                failure.append(int(idx))
            else:
                non_failure.append(int(idx))
        self._failure_starts = failure
        self._non_failure_starts = non_failure
        self._last_storage_len = len(storage)

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
        (label=0). Online ``update()`` maps replay labels to LPB expert
        targets via ``1 - label`` (head outputs expert-likeness logits).
        """
        if not demos:
            return
        for t in demos:
            self._base.add(t)
        self._last_storage_len = -1

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
                "Wait until at least one intervention frame is recorded "
                "before calling sample (gate with `ready(batch_size)`)."
            )
        if n_non_failure > 0 and not self._non_failure_starts:
            raise RuntimeError(
                "DiscriminatorReplayBuffer.sample: non_failure pool is empty. "
                "Call bootstrap_from_demos(...) or wait for a non-intervention "
                "rollout."
            )

        failure_idx = self._sample_indices(
            self._failure_starts, n_failure, allow_replacement=True
        )
        non_failure_idx = self._sample_indices(
            self._non_failure_starts,
            n_non_failure,
            allow_replacement=len(self._non_failure_starts) < n_non_failure,
        )
        all_idx = failure_idx + non_failure_idx
        camera_names = list(self._base.camera_names)

        images: list[np.ndarray] = []
        proprio: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        with self._base._lock:  # noqa: SLF001
            storage = self._base._storage  # noqa: SLF001
            for i in all_idx:
                t = storage[i]
                obs = t.obs
                images.append(_stack_views_uint8(obs, camera_names))
                proprio.append(np.asarray(obs["state"], dtype=np.float32))
                actions.append(np.asarray(t.action, dtype=np.float32))

        image_tensor = _to_image_tensor(np.stack(images, axis=0), device)
        proprio_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(proprio, axis=0))
        ).float()
        action_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(actions, axis=0))
        ).float()

        with torch.no_grad():
            context = self._encoder.encode(
                image_obs_raw=image_tensor,
                proprio_raw=proprio_tensor,
                action_real=action_tensor,
            )

        labels = torch.zeros(len(all_idx), dtype=torch.float32)
        # First `len(failure_idx)` rows correspond to intervention frames
        # → label=1 (failure / positive class).
        labels[: len(failure_idx)] = 1.0

        target_device = torch.device(device)
        return DiscriminatorBatch(
            context=context.to(target_device),
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
