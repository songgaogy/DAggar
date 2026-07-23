from __future__ import annotations

from typing import Any, Callable

from robosuite.pipeline.src.data.transitions import Transition

from .agent import AWRAgent
from .config import TrainerConfig


class AWRTrainer:
    """Synchronous episode-boundary AWR learner."""

    def __init__(
        self, agent: AWRAgent, config: TrainerConfig | None = None
    ) -> None:
        self.agent = agent
        self.config = config or agent.trainer_config
        self.total_env_steps = 0
        self.total_episodes = 0
        self.total_value_updates = 0
        self.total_actor_updates = 0
        self.total_value_warmup_updates = 0
        self.total_inference_syncs = 0
        self._offline_demo_episodes = 0
        self._offline_online_episodes = 0

    @staticmethod
    def _bootstrap(
        transitions: list[Transition],
        *,
        store: Callable[[Transition], None],
        source: str,
        namespace: str,
        start_episode: int,
    ) -> int:
        episode, step = int(start_episode), 0
        for transition in transitions:
            info = dict(transition.info or {})
            info.update(
                {
                    "episode_namespace": namespace,
                    "episode_index": episode,
                    "episode_step": step,
                }
            )
            store(
                Transition(
                    obs=transition.obs,
                    action=transition.action,
                    reward=float(transition.reward),
                    next_obs=transition.next_obs,
                    done=transition.done,
                    grasp_penalty=transition.grasp_penalty,
                    is_intervention=transition.is_intervention,
                    info=info,
                    reward_source=transition.reward_source or "offline",
                    demo_source=transition.demo_source or source,
                )
            )
            step += 1
            if transition.done:
                episode, step = episode + 1, 0
        return episode + int(step > 0)

    def bootstrap_demo_buffer(
        self,
        transitions: list[Transition],
        *,
        demo_source: str = "expert",
        episode_namespace: str = "expert",
    ) -> None:
        self._offline_demo_episodes = self._bootstrap(
            transitions,
            store=self.agent.store_demo_transition,
            source=demo_source,
            namespace=episode_namespace,
            start_episode=self._offline_demo_episodes,
        )

    def bootstrap_online_buffer(
        self,
        transitions: list[Transition],
        *,
        demo_source: str,
        episode_namespace: str,
    ) -> None:
        self._offline_online_episodes = self._bootstrap(
            transitions,
            store=self.agent.store_online_transition,
            source=demo_source,
            namespace=episode_namespace,
            start_episode=self._offline_online_episodes,
        )

    def record_transition(
        self,
        *,
        obs: Any,
        action: Any,
        next_obs: Any,
        done: bool,
        reward: float,
        grasp_penalty: float | None = None,
        is_intervention: bool = False,
        info: dict[str, Any] | None = None,
        reward_source: str = "env",
        demo_source: str | None = None,
        episode_index: int,
        episode_step: int,
        episode_namespace: str = "online",
    ) -> Transition:
        payload = dict(info or {})
        payload.update(
            {
                "episode_namespace": str(episode_namespace),
                "episode_index": int(episode_index),
                "episode_step": int(episode_step),
            }
        )
        transition = Transition(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
            grasp_penalty=grasp_penalty,
            is_intervention=bool(is_intervention),
            info=payload,
            reward_source=reward_source,
            demo_source=demo_source
            or ("intervention" if is_intervention else None),
        )
        self.agent.store_transition(transition)
        self.total_env_steps += 1
        return transition

    def pretrain_value(
        self, num_steps: int, batch_size: int | None = None
    ) -> list[dict[str, float]]:
        if not self.agent.ready_for_value_warmup():
            raise RuntimeError("Not enough AWR samples for value warmup.")
        results = []
        for _ in range(int(num_steps)):
            metrics = self.agent.update_value_only(batch_size)
            self.total_value_updates += 1
            self.total_value_warmup_updates += 1
            results.append(self._progress(metrics))
        return results

    def train_episode(
        self,
        *,
        updates: int | None = None,
        batch_size: int | None = None,
    ) -> list[dict[str, float]]:
        """Run the fixed synchronous update block after one completed episode."""
        self.total_episodes += 1
        if self.total_env_steps < self.config.warmup_steps:
            return []
        if not self.agent.ready_for_update():
            return []
        results = []
        requested = (
            self.config.updates_per_episode if updates is None else int(updates)
        )
        for _ in range(max(0, requested)):
            metrics = self.agent.update(batch_size)
            self.total_value_updates += 1
            self.total_actor_updates += 1
            synced = self._maybe_sync_inference()
            metrics["inference_synced"] = float(synced)
            results.append(self._progress(metrics))
        return results

    def _maybe_sync_inference(self) -> bool:
        interval = max(1, self.config.inference_sync_interval)
        if self.total_actor_updates % interval:
            return False
        self.agent.sync_inference_policy()
        self.total_inference_syncs += 1
        return True

    def _progress(self, metrics: dict[str, float]) -> dict[str, float]:
        result = dict(metrics)
        result.update(
            {
                "learner/env_steps": float(self.total_env_steps),
                "learner/episodes": float(self.total_episodes),
                "learner/value_updates": float(self.total_value_updates),
                "learner/actor_updates": float(self.total_actor_updates),
                "learner/value_warmup_updates": float(
                    self.total_value_warmup_updates
                ),
                "learner/inference_syncs": float(self.total_inference_syncs),
            }
        )
        return result

    def state_dict(self) -> dict[str, int]:
        return {
            "total_env_steps": self.total_env_steps,
            "total_episodes": self.total_episodes,
            "total_value_updates": self.total_value_updates,
            "total_actor_updates": self.total_actor_updates,
            "total_value_warmup_updates": self.total_value_warmup_updates,
            "total_inference_syncs": self.total_inference_syncs,
            "offline_demo_episodes": self._offline_demo_episodes,
            "offline_online_episodes": self._offline_online_episodes,
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        for name in self.state_dict():
            attribute = (
                f"_{name}" if name.startswith("offline_") else name
            )
            if name in state:
                setattr(self, attribute, int(state[name]))


__all__ = ["AWRTrainer"]
