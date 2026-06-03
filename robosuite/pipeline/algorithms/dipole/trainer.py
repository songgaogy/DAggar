from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import torch

from robosuite.pipeline.common.types import Transition

from .agent import DipoleAgent


if TYPE_CHECKING:
    from robosuite.pipeline.algorithms.discriminator.encoder import SharedFrozenEncoder
    from robosuite.pipeline.algorithms.discriminator.online_bce import (
        OnlineBCEDiscriminator,
    )
    from robosuite.pipeline.algorithms.discriminator.replay import (
        DiscriminatorReplayBuffer,
    )
    from robosuite.pipeline.algorithms.q_learning.iql import IQLLearner
    from robosuite.pipeline.algorithms.q_learning.replay import IQLReplayBuffer


logger = logging.getLogger(__name__)


class DipoleTrainer:
    def __init__(
        self,
        agent: DipoleAgent,
        config=None,
        *,
        iql_learner: "IQLLearner | None" = None,
        discriminator: "OnlineBCEDiscriminator | None" = None,
        iql_replay: "IQLReplayBuffer | None" = None,
        disc_replay: "DiscriminatorReplayBuffer | None" = None,
        shared_encoder: "SharedFrozenEncoder | None" = None,
        learner_device: str = "cuda:1",
        iql_batch_size: int = 64,
        disc_batch_size: int = 64,
        disc_update_freq: int = 1,
        iql_update_freq: int = 1,
    ) -> None:
        self.agent = agent
        self.config = config or agent.trainer_config
        self.total_env_steps = 0
        self.total_updates = 0
        self.total_publishes = 0
        self.last_published_update = 0
        self.total_pretrain_updates = 0
        self._offline_bootstrap_episodes = 0
        self._async_condition = threading.Condition()
        self._async_metrics: deque[dict[str, float]] = deque()
        self._async_thread: threading.Thread | None = None
        self._async_stop_requested = False
        self._async_pending_updates = 0
        self._async_pending_batch_size: int | None = None
        self._async_busy = False
        self._async_error: BaseException | None = None
        # DIPOLE-RL hooks (None == disabled, trainer degenerates to legacy DIPOLE)
        self.iql_learner = iql_learner
        self.discriminator = discriminator
        self.iql_replay = iql_replay
        self.disc_replay = disc_replay
        self.shared_encoder = shared_encoder
        self.learner_device = str(learner_device)
        self.iql_batch_size = int(iql_batch_size)
        self.disc_batch_size = int(disc_batch_size)
        self.disc_update_freq = int(disc_update_freq)
        self.iql_update_freq = int(iql_update_freq)

    def bootstrap_demo_buffer(
        self,
        transitions: list[Transition],
        demo_source: str = "offline_demo",
        *,
        mirror_online: bool = True,
    ) -> None:
        episode_index = int(self._offline_bootstrap_episodes)
        episode_step = 0
        for transition in transitions:
            if transition.reward is None:
                raise ValueError("Offline demo transitions must include reward values before bootstrapping.")
            info = {} if transition.info is None else dict(transition.info)
            info.setdefault("episode_index", int(episode_index))
            info.setdefault("episode_step", int(episode_step))
            info.setdefault("buffer_role", "offline")
            normalized = Transition(
                obs=transition.obs,
                action=transition.action,
                reward=transition.reward,
                next_obs=transition.next_obs,
                done=transition.done,
                grasp_penalty=transition.grasp_penalty,
                is_intervention=bool(transition.is_intervention),
                info=info,
                reward_source=transition.reward_source or "precomputed",
                demo_source=transition.demo_source or demo_source,
            )
            self.agent.store_demo_transition(normalized)
            if mirror_online:
                self.agent.store_online_transition(normalized)
            episode_step += 1
            if bool(transition.done):
                episode_index += 1
                episode_step = 0
        self._offline_bootstrap_episodes = int(episode_index if episode_step == 0 else episode_index + 1)

    def record_transition(
        self,
        *,
        obs,
        action,
        next_obs,
        done: bool,
        reward: float,
        grasp_penalty: float | None = None,
        is_intervention: bool = False,
        info: dict[str, Any] | None = None,
        reward_source: str | None = None,
        demo_source: str | None = None,
        episode_index: int | None = None,
        episode_step: int | None = None,
    ) -> Transition:
        info_payload = {} if info is None else dict(info)
        if episode_index is not None:
            info_payload.setdefault("episode_index", int(episode_index))
        if episode_step is not None:
            info_payload.setdefault("episode_step", int(episode_step))
        transition = Transition(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=done,
            grasp_penalty=grasp_penalty,
            is_intervention=is_intervention,
            info=info_payload,
            reward_source=reward_source or "env",
            demo_source=demo_source or ("intervention" if is_intervention else None),
        )
        self.agent.store_transition(transition)
        # DIPOLE-RL fanout: disc_replay reclassifies lazily but we still hand the
        # transition to it (currently a no-op) so future swaps remain symmetric.
        # iql_replay shares the agent.online_buffer by reference; no add needed.
        if self.disc_replay is not None:
            self.disc_replay.add_from_transition(transition)
        self.total_env_steps += 1
        return transition

    def pretrain(self, num_steps: int, batch_size: int | None = None) -> list[dict[str, float]]:
        metrics_list: list[dict[str, float]] = []
        for _ in range(int(num_steps)):
            metrics = self.train_step(batch_size=batch_size)
            self.total_pretrain_updates += 1
            published = self._maybe_publish_inference_policy()
            metrics_list.append(self._attach_progress_metrics(metrics, published=published))
        return metrics_list

    def train_step(self, batch_size: int | None = None) -> dict[str, float]:
        if not self.agent.ready_for_update(batch_size=batch_size):
            raise RuntimeError("Not enough valid dipole demo sequences are available for an update.")

        metrics: dict[str, float] = {}

        # 1. IQL Q + V (before flow so AdvantageG sees Q/V from this tick).
        if (
            self.iql_learner is not None
            and self.iql_replay is not None
            and (self.discriminator is None or self.shared_encoder is not None)
            and self.iql_replay.ready(self.iql_batch_size)
        ):
            from robosuite.pipeline.algorithms.q_learning.common import IQLActorBatch

            # Each iteration resamples and updates Q/V. metrics keep the last
            # iteration only (WandB x-axis is env step, not inner critic step).
            for _ in range(self.iql_update_freq):
                step_batch = self.iql_replay.sample_step_batch(
                    self.iql_batch_size,
                    encoder=self.shared_encoder,
                    discriminator=self.discriminator,
                    device=self.learner_device,
                )
                iql_metrics = self.iql_learner.update(step_batch)
                for k, v in iql_metrics.items():
                    metrics[f"iql/{k}"] = float(v)
                with torch.no_grad():
                    adv = self.iql_learner.compute_advantage_for_batch(
                        IQLActorBatch(
                            image_obs_raw=step_batch.image_obs_raw,
                            proprio_raw=step_batch.proprio_raw,
                            action_chunk_raw=step_batch.action_chunk,
                        )
                    )
                metrics["advantage_mean"] = float(adv.mean().item())
                if self.discriminator is not None:
                    meta = step_batch.metadata or {}
                    if "disc_reward_chunk_mean" in meta:
                        chunk_mean = float(meta["disc_reward_chunk_mean"])
                        metrics["disc_reward_chunk_mean"] = chunk_mean
                        # Legacy dashboards key (now = mean over all H frames).
                        metrics["disc_reward_mean"] = chunk_mean
                    if "disc_reward_first_frame_mean" in meta:
                        metrics["disc_reward_first_frame_mean"] = float(
                            meta["disc_reward_first_frame_mean"]
                        )

        # 2. Discriminator BCE head (before flow so disc logit in G is current).
        if (
            self.discriminator is not None
            and self.disc_replay is not None
            and self.disc_replay.ready(self.disc_batch_size)
        ):
            for _ in range(self.disc_update_freq):
                disc_batch = self.disc_replay.sample(self.disc_batch_size, device=self.learner_device)
                disc_metrics = self.discriminator.update(disc_batch)
                for k, v in disc_metrics.items():
                    metrics[f"disc/{k}"] = float(v)

        # 3. Flow / policy update last (online:demo 1:1 batch; uses g_provider from steps 1–2).
        batch = self.agent.sample_training_batch(batch_size=batch_size)
        flow_metrics = dict(self.agent.update(batch=batch))
        metrics.update(flow_metrics)

        metrics["iql_update_freq"] = float(self.iql_update_freq)
        metrics["disc_update_freq"] = float(self.disc_update_freq)
        self.total_updates += 1
        return metrics

    # ------------------------------------------------------------------ #
    # IQL warmup helpers (called by train_dipole_rl.py before rollout)   #
    # ------------------------------------------------------------------ #

    def pretrain_iql_value(self, num_steps: int) -> list[dict[str, float]]:
        """V-only IQL warmup. Disc is treated as frozen here (no `.update()` is
        ever invoked during this loop); we pass it through so r_disc enters the
        reward composition consistently with the online phase."""
        if self.iql_learner is None or self.iql_replay is None:
            raise RuntimeError("pretrain_iql_value requires iql_learner / iql_replay.")
        out: list[dict[str, float]] = []
        for _ in range(int(num_steps)):
            if not self.iql_replay.ready(self.iql_batch_size):
                logger.warning("IQL warmup_value: replay not ready; stopping early.")
                break
            step_batch = self.iql_replay.sample_step_batch(
                self.iql_batch_size,
                encoder=self.shared_encoder,
                discriminator=self.discriminator,
                device=self.learner_device,
            )
            m = self.iql_learner.warmup_value_only(step_batch)
            out.append({f"iql_warmup_v/{k}": float(v) for k, v in m.items()})
        return out

    def pretrain_iql_full(self, num_steps: int) -> list[dict[str, float]]:
        """Full IQL warmup (Q + V + target polyak). Disc is frozen here — see
        `pretrain_iql_value` docstring."""
        if self.iql_learner is None or self.iql_replay is None:
            raise RuntimeError("pretrain_iql_full requires iql_learner / iql_replay.")
        out: list[dict[str, float]] = []
        for _ in range(int(num_steps)):
            if not self.iql_replay.ready(self.iql_batch_size):
                logger.warning("IQL warmup_full: replay not ready; stopping early.")
                break
            step_batch = self.iql_replay.sample_step_batch(
                self.iql_batch_size,
                encoder=self.shared_encoder,
                discriminator=self.discriminator,
                device=self.learner_device,
            )
            m = self.iql_learner.update(step_batch)
            out.append({f"iql_warmup_full/{k}": float(v) for k, v in m.items()})
        return out

    def maybe_update(self, *, env_step: int | None = None, batch_size: int | None = None) -> list[dict[str, float]]:
        self._raise_async_error()
        current_step = self.total_env_steps if env_step is None else int(env_step)
        if current_step < int(self.config.warmup_steps):
            return self.drain_async_metrics()
        if not self.agent.ready_for_update(batch_size=batch_size):
            return self.drain_async_metrics()
        metrics_list = []
        for _ in range(int(self.config.updates_per_step)):
            metrics = self.train_step(batch_size=batch_size)
            published = self._maybe_publish_inference_policy()
            metrics_list.append(self._attach_progress_metrics(metrics, published=published))
        return metrics_list

    def maybe_update_async(self, *, env_step: int | None = None, batch_size: int | None = None) -> list[dict[str, float]]:
        self._raise_async_error()
        current_step = self.total_env_steps if env_step is None else int(env_step)
        if current_step >= int(self.config.warmup_steps) and self.agent.ready_for_update(batch_size=batch_size):
            self.start_async_worker()
            with self._async_condition:
                requested = int(self.config.updates_per_step)
                max_pending = max(1, int(self.config.max_pending_updates))
                in_flight = int(self._async_pending_updates) + (1 if self._async_busy else 0)
                accepted = min(requested, max(0, max_pending - in_flight))
                skipped = requested - accepted
                if accepted > 0:
                    self._async_pending_updates += accepted
                    self._async_pending_batch_size = None if batch_size is None else int(batch_size)
                    self._async_condition.notify_all()
                if skipped > 0:
                    logger.debug(
                        "Skipped %d async DIPOLE learner update(s): pending=%d busy=%s max_pending=%d",
                        skipped,
                        int(self._async_pending_updates),
                        bool(self._async_busy),
                        max_pending,
                    )
        return self.drain_async_metrics()

    def start_async_worker(self) -> None:
        self._raise_async_error()
        with self._async_condition:
            if self._async_thread is not None:
                return
            self._async_stop_requested = False
            self._async_thread = threading.Thread(
                target=self._async_update_loop, name="dipole_learner", daemon=True,
            )
            self._async_thread.start()

    def pending_async_updates(self) -> int:
        with self._async_condition:
            return int(self._async_pending_updates) + int(self._async_busy)

    def drain_async_metrics(self) -> list[dict[str, float]]:
        self._raise_async_error()
        with self._async_condition:
            metrics = list(self._async_metrics)
            self._async_metrics.clear()
        return metrics

    def flush_async_updates(self, timeout: float | None = None) -> None:
        self._raise_async_error()
        with self._async_condition:
            if self._async_thread is None:
                return
            if timeout is None:
                while self._async_pending_updates > 0 or self._async_busy:
                    self._async_condition.wait(timeout=0.1)
                    self._raise_async_error()
                return

        with self._async_condition:
            end_time = time.monotonic() + float(timeout)
            remaining_timeout = timeout
            while self._async_pending_updates > 0 or self._async_busy:
                self._raise_async_error()
                if remaining_timeout is not None and remaining_timeout <= 0.0:
                    raise TimeoutError("Timed out while waiting for async learner updates to finish.")
                wait_timeout = 0.1 if remaining_timeout is None else min(0.1, remaining_timeout)
                self._async_condition.wait(timeout=wait_timeout)
                if remaining_timeout is not None:
                    remaining_timeout = end_time - time.monotonic()

    def close_async_worker(self, wait: bool = True) -> None:
        if wait:
            self.flush_async_updates()
        with self._async_condition:
            thread = self._async_thread
            if thread is None:
                return
            self._async_stop_requested = True
            self._async_condition.notify_all()
        thread.join(timeout=5.0)
        if thread.is_alive():
            raise RuntimeError("Async learner worker did not stop cleanly.")
        with self._async_condition:
            if self._async_thread is thread:
                self._async_thread = None
        self._raise_async_error()

    def state_dict(self) -> dict[str, int]:
        return {
            "total_env_steps": int(self.total_env_steps),
            "total_updates": int(self.total_updates),
            "total_publishes": int(self.total_publishes),
            "last_published_update": int(self.last_published_update),
            "total_pretrain_updates": int(self.total_pretrain_updates),
            "offline_bootstrap_episodes": int(self._offline_bootstrap_episodes),
        }

    def load_state_dict(self, state_dict: dict[str, Any] | None) -> None:
        if not state_dict:
            return
        self.total_env_steps = int(state_dict.get("total_env_steps", self.total_env_steps))
        self.total_updates = int(state_dict.get("total_updates", self.total_updates))
        self.total_publishes = int(state_dict.get("total_publishes", self.total_publishes))
        self.last_published_update = int(state_dict.get("last_published_update", self.last_published_update))
        self.total_pretrain_updates = int(state_dict.get("total_pretrain_updates", self.total_pretrain_updates))
        self._offline_bootstrap_episodes = int(
            state_dict.get("offline_bootstrap_episodes", self._offline_bootstrap_episodes)
        )

    def progress_snapshot(self) -> dict[str, int]:
        return {
            "env_steps": int(self.total_env_steps),
            "total_updates": int(self.total_updates),
            "actor_updates": int(self.total_updates),
            "publish_count": int(self.total_publishes),
            "last_published_update": int(self.last_published_update),
            "updates_until_publish": int(self.updates_until_next_publish()),
            "total_pretrain_updates": int(self.total_pretrain_updates),
        }

    def updates_until_next_publish(self) -> int:
        publish_interval = max(1, int(self.config.steps_per_update))
        remainder = int(self.total_updates) % publish_interval
        if remainder == 0:
            return publish_interval
        return publish_interval - remainder

    def _maybe_publish_inference_policy(self) -> bool:
        publish_interval = max(1, int(self.config.steps_per_update))
        if self.total_updates % publish_interval != 0:
            return False
        self.agent.sync_inference_policy()
        self.total_publishes += 1
        self.last_published_update = int(self.total_updates)
        return True

    def _attach_progress_metrics(self, metrics: dict[str, float], *, published: bool) -> dict[str, float]:
        progress = self.progress_snapshot()
        summarized_metrics = dict(metrics)
        summarized_metrics["learner_total_updates"] = float(progress["total_updates"])
        summarized_metrics["learner_actor_updates"] = float(progress["actor_updates"])
        summarized_metrics["learner_publish_count"] = float(progress["publish_count"])
        summarized_metrics["learner_last_published_update"] = float(progress["last_published_update"])
        summarized_metrics["learner_updates_until_publish"] = float(progress["updates_until_publish"])
        summarized_metrics["learner_total_pretrain_updates"] = float(progress["total_pretrain_updates"])
        summarized_metrics["learner_published"] = 1.0 if published else 0.0
        return summarized_metrics

    def _async_update_loop(self) -> None:
        while True:
            with self._async_condition:
                while self._async_pending_updates == 0 and not self._async_stop_requested:
                    self._async_condition.wait(timeout=0.1)
                if self._async_stop_requested and self._async_pending_updates == 0:
                    self._async_thread = None
                    self._async_condition.notify_all()
                    return
                self._async_pending_updates -= 1
                batch_size = self._async_pending_batch_size
                self._async_busy = True

            try:
                metrics = self.train_step(batch_size=batch_size)
                published = self._maybe_publish_inference_policy()
                with self._async_condition:
                    self._async_metrics.append(self._attach_progress_metrics(metrics, published=published))
            except BaseException as exc:
                with self._async_condition:
                    self._async_error = exc
                    self._async_stop_requested = True
                    self._async_condition.notify_all()
                return
            finally:
                with self._async_condition:
                    self._async_busy = False
                    self._async_condition.notify_all()

    def _raise_async_error(self) -> None:
        if self._async_error is not None:
            raise RuntimeError(f"Async dipole learner failed: {self._async_error}") from self._async_error
