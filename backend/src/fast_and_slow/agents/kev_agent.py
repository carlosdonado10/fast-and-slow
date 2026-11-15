"""Kev arms: zero-shot, in-context (rules + worked examples in the state), and any RL-tuned policy at eval time."""

from __future__ import annotations

import time

import numpy as np
import torch

from ..config import WorldConfig
from ..notes.bank import NoteBank
from ..sim.desk import Episode, Observation
from ..sim.render import ORDER_Q, REGIME_Q, request
from ..sim.world import sample_scenario
from .base import Agent, Decision
from .baselines import BayesAgent
from .policy import DecisionPolicy


class KevAgent(Agent):
    def __init__(
        self,
        name: str,
        world: WorldConfig,
        policy: DecisionPolicy,
        context: str | None = None,
        ask_regime: bool = True,
        batch_size: int = 32,
    ):
        self.name, self.w, self.policy, self.context = name, world, policy, context
        self.ask_regime, self.batch_size = ask_regime, batch_size

    def requests(self, obs: list[Observation]) -> list[dict[str, object]]:
        reqs = [request(self.w, o, self.context) for o in obs]
        if not self.ask_regime:
            for r in reqs:
                r["questions"] = {ORDER_Q: r["questions"][ORDER_Q]}
        return reqs

    def act_batch(self, obs: list[Observation], live: list[int]) -> list[Decision]:
        out: list[Decision] = []
        for i in range(0, len(obs), self.batch_size):
            chunk = obs[i : i + self.batch_size]
            t0 = time.perf_counter()
            scores = self.policy.score(self.requests(chunk))
            per = (time.perf_counter() - t0) * 1000 / len(chunk)
            for s in scores:
                p = torch.softmax(s[ORDER_Q], -1).cpu().numpy().astype(float)
                rp = torch.softmax(s[REGIME_Q], -1).cpu().numpy().astype(float) if REGIME_Q in s else None
                out.append(Decision(int(np.argmax(p)), p, rp, per))
        return out


def icl_context(world: WorldConfig, bank: NoteBank, horizon: int, k: int, seed: int = 123) -> str:
    """k worked examples from an experienced planner's log (Bayes decisions on *training* scenarios and wording).

    Picks examples that show different situations: notes from each source, a distractor, and a no-note week where possible.
    These k decisions count as experience the arm consumed (the learning curve plots the arm at x = k).
    """
    agent = BayesAgent(world, max(horizon, 2))
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    seen_kinds: set[str] = set()
    for s in rng.integers(0, 5_000, size=200):
        if len(picked) >= k:
            break
        sc = sample_scenario(world, max(horizon, 2), int(s))
        ep = Episode(world, sc, bank.texts(sc, "train"))
        agent.begin([sc])
        while not ep.done and len(picked) < k:
            o = ep.observe()
            d = agent.act(o, 0)
            note = o.this_week_note
            kind = note.note.cell.split(":")[0] if note.note.kind == "signal" else note.note.kind
            out = ep.step(world.order_menu[d.action])
            if kind in seen_kinds and len(seen_kinds) < 5:
                continue
            seen_kinds.add(kind)
            inbox = f'"{note.text}"' if note.text else "nothing new"
            picked.append(
                f"- Week {o.week + 1} of {o.horizon}, {o.inventory} units on hand, inbox: {inbox} -> ordered {out.order}; "
                f"demand turned out {out.demand}, profit that week ${out.reward:.0f}."
            )
    return "Examples from an experienced planner's log on other weeks:\n" + "\n".join(picked)
