from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .agent import DSRLAgent
from .batch import DSRLBatch

BatchProvider = Callable[[int], DSRLBatch]


class DSRLTrainer:
    def __init__(self, agent: DSRLAgent, batch_provider: BatchProvider) -> None:
        self.agent = agent
        self.batch_provider = batch_provider
        self.cycles = 0

    def update_cycle(self) -> dict[str, float]:
        metrics: dict[str, list[float]] = defaultdict(list)
        for _ in range(self.agent.config.utd_steps):
            result = self.agent.update(self.batch_provider(self.agent.config.batch_size))
            for name, value in result.items():
                metrics[name].append(value)
        self.cycles += 1
        summarized = {name: sum(values) / len(values) for name, values in metrics.items()}
        summarized.update(
            {
                "qa_updates_this_cycle": float(self.agent.config.utd_steps),
                "actor_updates_this_cycle": float(self.agent.config.utd_steps),
                "alpha_updates_this_cycle": float(self.agent.config.utd_steps),
                "total_qa_updates": float(self.agent.qa_updates),
                "total_actor_updates": float(self.agent.actor_updates),
                "total_alpha_updates": float(self.agent.alpha_updates),
            }
        )
        return summarized

    def state_dict(self) -> dict[str, Any]:
        return {"cycles": self.cycles, "agent": self.agent.state_dict()}

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> None:
        self.agent.load_state_dict(state["agent"], strict=strict)
        self.cycles = int(state.get("cycles", 0))
