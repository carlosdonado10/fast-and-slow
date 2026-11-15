"""The replenishment desk: hidden regimes, demand, notes and the economics of one period.

Everything exogenous (the regime path, demand draws, which notes arrive and what they claim) is drawn up front from the
scenario seed. Only inventory depends on the agent. So two rollouts on the same seed face exactly the same world, which
is what makes group-relative advantages (Kev RL) and paired comparisons between arms fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.stats import poisson

from ..config import WorldConfig

NONE, DISTRACTOR, SIGNAL = "none", "distractor", "signal"
CLAIMS = ("down", "flat", "up")  # claim k asserts regime k


@dataclass(frozen=True)
class Note:
    kind: str  # none | distractor | signal
    source: int = -1  # index into WorldConfig.sources
    claim: int = -1  # index into CLAIMS (== regime index it asserts)
    variant: int = 0  # which paraphrase of the cell to show (resolved against the note bank's split)

    @property
    def cell(self) -> str:
        if self.kind == SIGNAL:
            return f"{self.source}:{self.claim}"
        return self.kind


@dataclass(frozen=True)
class Scenario:
    seed: int
    horizon: int
    regimes: tuple[int, ...]
    demands: tuple[int, ...]
    notes: tuple[Note, ...]


@dataclass
class PeriodOutcome:
    order: int
    demand: int
    sales: int
    lost: int
    inventory_end: int
    reward: float
    breakdown: dict[str, float] = field(default_factory=dict)


def transition_matrix(w: WorldConfig) -> np.ndarray:
    k = len(w.regimes)
    off = (1 - w.stay_prob) / (k - 1)
    return np.full((k, k), off) + np.eye(k) * (w.stay_prob - off)


@lru_cache(maxsize=8)
def _demand_pmf(means: tuple[float, ...], max_demand: int) -> np.ndarray:
    d = np.arange(max_demand + 1)
    pmf = np.stack([poisson.pmf(d, m) for m in means])
    pmf[:, -1] += 1 - pmf.sum(axis=1)  # lump the tail into the cap
    return pmf


def demand_pmf(w: WorldConfig) -> np.ndarray:
    """[regimes, max_demand + 1]"""
    return _demand_pmf(tuple(w.demand_means), w.max_demand)


def note_likelihood(w: WorldConfig, note: Note) -> np.ndarray:
    """P(note | regime) for each regime. Notes without information are flat (no update)."""
    k = len(w.regimes)
    if note.kind != SIGNAL:
        return np.ones(k)
    acc = w.source_accuracy[note.source]
    lik = np.full(k, (1 - acc) / (k - 1))
    lik[note.claim] = acc
    return lik


def note_outcomes(w: WorldConfig) -> list[tuple[Note, np.ndarray]]:
    """Every distinguishable note event with its probability under each regime: [(note, P(note | regime))]."""
    k = len(w.regimes)
    out = [(Note(NONE), np.full(k, 1 - w.note_prob)), (Note(DISTRACTOR), np.full(k, w.note_prob * w.distractor_prob))]
    p_sig = w.note_prob * (1 - w.distractor_prob)
    for s, mix in enumerate(w.source_mix):
        for c in range(k):
            out.append((Note(SIGNAL, s, c), p_sig * mix * note_likelihood(w, Note(SIGNAL, s, c))))
    return out


def sample_scenario(w: WorldConfig, horizon: int, seed: int, variants_per_cell: int = 1_000_000) -> Scenario:
    rng = np.random.default_rng(seed)
    k = len(w.regimes)
    T = transition_matrix(w)
    pmf = demand_pmf(w)
    z = int(rng.choice(k, p=np.asarray(w.regime_prior)))
    regimes, demands, notes = [], [], []
    for t in range(horizon):
        if t > 0:
            z = int(rng.choice(k, p=T[z]))
        regimes.append(z)
        demands.append(int(rng.choice(w.max_demand + 1, p=pmf[z])))
        variant = int(rng.integers(variants_per_cell))
        u = rng.random()
        if u >= w.note_prob:
            notes.append(Note(NONE, variant=variant))
        elif rng.random() < w.distractor_prob:
            notes.append(Note(DISTRACTOR, variant=variant))
        else:
            s = int(rng.choice(len(w.sources), p=np.asarray(w.source_mix)))
            correct = rng.random() < w.source_accuracy[s]
            claim = z if correct else int(rng.choice([c for c in range(k) if c != z]))
            notes.append(Note(SIGNAL, s, claim, variant))
    return Scenario(seed, horizon, tuple(regimes), tuple(demands), tuple(notes))


def clip_order(w: WorldConfig, inventory: int, order: int) -> int:
    return max(0, min(order, w.inventory_cap - inventory))


def step_economics(w: WorldConfig, inventory: int, order: int, demand: int, last: bool) -> PeriodOutcome:
    """One period: the order arrives, demand happens, unmet demand is lost, leftovers carry (or are salvaged at the end)."""
    q = clip_order(w, inventory, order)
    on_hand = inventory + q
    sales = min(on_hand, demand)
    lost = demand - sales
    left = on_hand - sales
    parts = {
        "revenue": w.price * sales,
        "purchase": -w.unit_cost * q,
        "fixed": -w.fixed_order_cost if q > 0 else 0.0,
        "holding": -w.holding_cost * left,
        "stockout": -w.stockout_penalty * lost,
        "salvage": w.salvage_value * left if last else 0.0,
    }
    return PeriodOutcome(q, demand, sales, lost, 0 if last else left, float(sum(parts.values())), parts)
