"""Reference arms: random (0%), Bayes-optimal (100%), perfect information (upper bound), base-stock (today's tool)."""

from __future__ import annotations

import itertools

import numpy as np

from ..config import OracleConfig, WorldConfig
from ..oracle.bayes import BayesPlanner, PerfectInfoPlanner
from ..sim.desk import Observation
from ..sim.world import sample_scenario, step_economics
from .base import Agent, Decision, one_hot


def bayes_belief(planner: BayesPlanner, obs: Observation) -> np.ndarray:
    """Exact posterior over this week's regime, rebuilt from everything observed so far (notes are read perfectly)."""
    b = planner.prior()
    for t in range(obs.week + 1):
        b = planner.after_note(b, obs.notes[t].note)
        if t < obs.week:
            b = planner.after_demand(b, obs.history[t].demand)
    return b


class RandomAgent(Agent):
    name = "random"

    def __init__(self, world: WorldConfig, seed: int = 0):
        self.A = len(world.order_menu)
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Observation, scenario_index: int) -> Decision:
        return Decision(int(self.rng.integers(self.A)), np.full(self.A, 1 / self.A))


class BayesAgent(Agent):
    name = "bayes"
    privileged = True  # reads note attributes directly: a perfect parser

    def __init__(self, world: WorldConfig, horizon: int, cfg: OracleConfig | None = None):
        self.planner = BayesPlanner(world, horizon, cfg)

    def q_and_belief(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        b = bayes_belief(self.planner, obs)
        return self.planner.q_values(obs.week, obs.inventory, b), b

    def act(self, obs: Observation, scenario_index: int) -> Decision:
        q, b = self.q_and_belief(obs)
        a = int(np.argmax(q))
        return Decision(a, one_hot(a, len(q)), b)


class OracleAgent(Agent):
    name = "oracle"
    privileged = True  # sees the hidden regime

    def __init__(self, world: WorldConfig, horizon: int):
        self.pi = PerfectInfoPlanner(world, horizon)
        self.K = len(world.regimes)

    def act(self, obs: Observation, scenario_index: int) -> Decision:
        z = self._scenarios[scenario_index].regimes[obs.week]
        a = self.pi.best_action(obs.week, obs.inventory, z)
        return Decision(a, one_hot(a, len(self.pi.Q[0][0, 0])), one_hot(z, self.K))


class BaseStockAgent(Agent):
    """(s, S) policy on numbers only: if stock < s, order up to S (nearest menu option). Tuned on training scenarios."""

    name = "basestock"

    def __init__(self, world: WorldConfig, s: int, S: int):
        self.w, self.s, self.S = world, s, S
        self.menu = np.asarray(world.order_menu)

    def act(self, obs: Observation, scenario_index: int) -> Decision:
        need = self.S - obs.inventory if obs.inventory < self.s else 0
        a = int(np.argmin(np.abs(self.menu - need)))
        return Decision(a, one_hot(a, len(self.menu)))

    @classmethod
    def tune(cls, world: WorldConfig, horizon: int, seeds: list[int]) -> BaseStockAgent:
        scen = [sample_scenario(world, horizon, s) for s in seeds]
        menu = np.asarray(world.order_menu)
        best, best_v = (0, 0), -np.inf
        for s, S in itertools.product(range(0, 31, 2), range(4, 51, 2)):
            if s > S:
                continue
            total = 0.0
            for sc in scen:
                inv = world.start_inventory
                for t in range(horizon):
                    need = S - inv if inv < s else 0
                    q = int(menu[np.argmin(np.abs(menu - need))])
                    out = step_economics(world, inv, q, sc.demands[t], last=t == horizon - 1)
                    inv, total = out.inventory_end, total + out.reward
            if total > best_v:
                best, best_v = (s, S), total
        return cls(world, *best)
