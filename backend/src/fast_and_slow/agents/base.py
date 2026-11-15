"""The one interface every arm implements, so the evaluator treats Bayes, Kev and PPO identically."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.desk import Observation
from ..sim.world import Scenario


@dataclass
class Decision:
    action: int  # index into WorldConfig.order_menu
    probs: np.ndarray  # [A] the agent's distribution over the order menu (one-hot for deterministic agents)
    regime_probs: np.ndarray | None = None  # [K] the agent's belief about this week's regime, if it states one
    latency_ms: float = 0.0

    @property
    def confidence(self) -> float:
        return float(self.probs[self.action])


class Agent:
    name: str = "agent"
    #: reference arms may look at privileged information (note attributes, the hidden regime); learners never do
    privileged: bool = False

    def begin(self, scenarios: list[Scenario]) -> None:
        """Called once before a batch of episodes runs in lockstep (index i of every act_batch call = scenario i)."""
        self._scenarios = scenarios

    def act_batch(self, obs: list[Observation], live: list[int]) -> list[Decision]:
        """obs[j] belongs to scenario live[j]. Default: one call per observation."""
        return [self.act(o, i) for o, i in zip(obs, live, strict=True)]

    def act(self, obs: Observation, scenario_index: int) -> Decision:  # pragma: no cover - interface
        raise NotImplementedError


def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v
