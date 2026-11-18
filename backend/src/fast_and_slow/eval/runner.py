"""Run any agent on a fixed scenario set, in lockstep, and record every decision with its ground-truth labels.

Labels come from the Bayes planner and the perfect-information planner (never shown to the agent):
- acceptable: the chosen order is within the regret tolerance of the Bayes-optimal one
- decidable: the Bayes action is the full-information best action under every plausible regime (a dominant strategy)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..agents.base import Agent
from ..agents.baselines import bayes_belief
from ..config import OracleConfig, WorldConfig
from ..notes.bank import NoteBank
from ..oracle.bayes import BayesPlanner, PerfectInfoPlanner, is_decidable
from ..progress import Progress
from ..sim.desk import Episode
from ..sim.world import Scenario, sample_scenario


@dataclass
class DecisionRecord:
    seed: int
    week: int
    inventory: int
    action: int
    confidence: float
    acceptable: bool
    decidable: bool
    bayes_action: int
    regret: float  # Bayes Q(best) - Q(chosen)
    true_regime: int
    bayes_posterior: list[float]
    regime_probs: list[float] | None
    latency_ms: float
    reward: float


@dataclass
class EvalResult:
    agent: str
    horizon: int
    returns: list[float]
    decisions: list[DecisionRecord] = field(default_factory=list)
    wall_seconds: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "horizon": self.horizon,
            "returns": self.returns,
            "wall_seconds": self.wall_seconds,
            "decisions": [asdict(d) for d in self.decisions],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> EvalResult:
        return cls(
            d["agent"], d["horizon"], d["returns"], [DecisionRecord(**x) for x in d["decisions"]], d["wall_seconds"]
        )


class Labeller:
    """Bayes + perfect-information planners for one horizon, shared across every agent evaluated at that horizon."""

    def __init__(self, world: WorldConfig, horizon: int, cfg: OracleConfig):
        self.w, self.cfg = world, cfg
        self.bayes = BayesPlanner(world, horizon, cfg)
        self.pi = PerfectInfoPlanner(world, horizon)


def eval_scenarios(world: WorldConfig, horizon: int, n: int, first_seed: int) -> list[Scenario]:
    return [sample_scenario(world, horizon, first_seed + i) for i in range(n)]


def evaluate(
    agent: Agent,
    world: WorldConfig,
    bank: NoteBank,
    scenarios: list[Scenario],
    labeller: Labeller,
    split: str = "test",
    record: bool = True,
    label: str | None = None,
) -> EvalResult:
    """label: print progress lines under this name (for the long, final evaluations; curve evals stay quiet)."""
    started = time.perf_counter()
    prog = Progress(label, sum(sc.horizon for sc in scenarios), "decisions") if label else None
    eps = [Episode(world, sc, bank.texts(sc, split)) for sc in scenarios]
    agent.begin(scenarios)
    recs: list[DecisionRecord] = []
    while True:
        live = [i for i, e in enumerate(eps) if not e.done]
        if not live:
            break
        obs = [eps[i].observe() for i in live]
        t0 = time.perf_counter()
        decisions = agent.act_batch(obs, live)
        per = (time.perf_counter() - t0) * 1000 / len(live)
        if prog:
            prog.advance(len(live))
        for i, o, d in zip(live, obs, decisions, strict=True):
            out = eps[i].step(world.order_menu[d.action])
            if record:
                b = bayes_belief(labeller.bayes, o)
                q = labeller.bayes.q_values(o.week, o.inventory, b)
                ba = int(np.argmax(q))
                recs.append(
                    DecisionRecord(
                        seed=scenarios[i].seed,
                        week=o.week,
                        inventory=o.inventory,
                        action=d.action,
                        confidence=d.confidence,
                        acceptable=bool(labeller.bayes.acceptable(q)[d.action]),
                        decidable=is_decidable(
                            ba, b, o.week, o.inventory, labeller.pi, labeller.cfg.decidable_posterior_floor
                        ),
                        bayes_action=ba,
                        regret=float(q.max() - q[d.action]),
                        true_regime=scenarios[i].regimes[o.week],
                        bayes_posterior=[round(float(x), 4) for x in b],
                        regime_probs=None if d.regime_probs is None else [round(float(x), 4) for x in d.regime_probs],
                        latency_ms=d.latency_ms or per,
                        reward=out.reward,
                    )
                )
    return EvalResult(agent.name, scenarios[0].horizon, [e.total for e in eps], recs, time.perf_counter() - started)
