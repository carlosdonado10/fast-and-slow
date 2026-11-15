"""Every number the report shows, computed from evaluation records and learning curves (docs/spec.md section 4)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .runner import DecisionRecord

# ---------------------------------------------------------------- returns


def pct_of_optimal(mean_return: float, ref_random: float, ref_bayes: float) -> float:
    """Practitioner scale: random = 0, Bayes-optimal = 100."""
    return 100.0 * (mean_return - ref_random) / (ref_bayes - ref_random)


def inbox_value(mean_return: float, ref_basestock: float, ref_bayes: float) -> float:
    """Planner scale: the numbers-only tool = 0, Bayes = 100. 'How much of what the inbox is worth did you capture?'"""
    return 100.0 * (mean_return - ref_basestock) / (ref_bayes - ref_basestock)


def bootstrap_ci(values: Sequence[float], n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        m = float(v.mean()) if len(v) else float("nan")
        return m, m
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------- calibration


@dataclass
class Reliability:
    bins: list[dict[str, float]]  # per bin: lo, hi, mean confidence, accuracy, count
    ece: float
    brier: float


def reliability(confidence: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> Reliability:
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in itertools.pairwise(edges):
        m = (c >= lo) & ((c < hi) if hi < 1 else (c <= hi))
        if not m.any():
            continue
        conf, acc = float(c[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        bins.append({"lo": float(lo), "hi": float(hi), "confidence": conf, "accuracy": acc, "count": int(m.sum())})
    brier = float(np.mean((c - y) ** 2)) if len(c) else float("nan")
    return Reliability(bins, float(ece), brier)


def multiclass_brier(probs: ArrayLike, target: ArrayLike) -> float:
    """Brier score of predicted distributions against a target distribution (one-hot truth or a posterior)."""
    p, t = np.asarray(probs, dtype=float), np.asarray(target, dtype=float)
    return float(np.mean(np.sum((p - t) ** 2, axis=1)))


def regime_calibration(records: Sequence[DecisionRecord], k: int = 3) -> dict[str, float] | None:
    """How honest is the agent's stated belief about the hidden regime? vs the truth, and vs the Bayes posterior."""
    rs = [r for r in records if r.regime_probs is not None]
    if not rs:
        return None
    probs = [r.regime_probs for r in rs if r.regime_probs is not None]
    truth = np.eye(k)[[r.true_regime for r in rs]]
    post = [r.bayes_posterior for r in rs]
    return {
        "brier_vs_truth": multiclass_brier(probs, truth),
        "brier_vs_bayes": multiclass_brier(probs, post),
        "bayes_brier_vs_truth": multiclass_brier(post, truth),
    }


# ---------------------------------------------------------------- hand-off


def automation_at_budget(confidence: Sequence[float], correct: Sequence[bool], budget: float) -> float:
    """Largest share of decisions the agent can take, most-confident first, keeping the error rate within budget."""
    c = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=bool)
    if len(c) == 0:
        return 0.0
    order = np.argsort(-c, kind="stable")
    errors = np.cumsum(~y[order])
    taken = np.arange(1, len(c) + 1)
    ok = errors / taken <= budget + 1e-12
    if not ok.any():
        return 0.0
    return float(taken[np.flatnonzero(ok)[-1]] / len(c))


def handoff(records: Sequence[DecisionRecord], budgets: Sequence[float] = (0.01, 0.02, 0.05, 0.1)) -> dict[str, object]:
    """Automation curves overall and by decision type; with the share of taken decisions that are decidable."""
    out: dict[str, object] = {}
    groups = {"all": list(records), "decidable": [r for r in records if r.decidable]}
    groups["judgement"] = [r for r in records if not r.decidable]
    for name, rs in groups.items():
        out[name] = {
            "n": len(rs),
            "accuracy": float(np.mean([r.acceptable for r in rs])) if rs else float("nan"),
            "automation": {
                f"{b:.2f}": automation_at_budget([r.confidence for r in rs], [r.acceptable for r in rs], b)
                for b in budgets
            },
        }
    return out


# ---------------------------------------------------------------- sample efficiency


@dataclass
class CurvePoint:
    decisions: int
    score: float  # % of optimal
    seconds: float = 0.0


def decisions_to_threshold(curve: Sequence[CurvePoint], threshold: float) -> float:
    """Training decisions until the score first reaches the threshold, interpolated in log-decisions.

    0 when the very first point already clears it (a zero-shot start); inf when it never does.
    """
    pts = sorted(curve, key=lambda p: p.decisions)
    if not pts:
        return float("inf")
    if pts[0].score >= threshold:
        return float(pts[0].decisions)
    for a, b in itertools.pairwise(pts):
        if b.score >= threshold:
            la, lb = np.log10(max(a.decisions, 1)), np.log10(max(b.decisions, 1))
            f = (threshold - a.score) / (b.score - a.score)
            return float(10 ** (la + f * (lb - la)))
    return float("inf")


def gain_per_decade(curve: Sequence[CurvePoint]) -> float:
    """Score gained per 10x more training decisions: least-squares slope of score on log10(decisions), x >= 1."""
    pts = [p for p in curve if p.decisions >= 1]
    if len(pts) < 2:
        return float("nan")
    x = np.log10([p.decisions for p in pts])
    y = np.asarray([p.score for p in pts])
    if np.ptp(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def area_under_curve(
    curve: Sequence[CurvePoint], lo: float, hi: float, floor: float = 0.0, ceil: float = 100.0
) -> float:
    """Area under the learning curve over log10 decisions in [lo, hi], scores clipped to [floor, ceil], scaled to [0, 1].

    Before its first point a curve is held at that point's score (Kev's zero-shot start is a real score at 0 decisions);
    after its last point it is held at the last score (no extrapolated gains).
    """
    pts = sorted(curve, key=lambda p: p.decisions)
    if not pts or hi <= lo:
        return float("nan")
    grid = np.linspace(np.log10(lo), np.log10(hi), 200)
    xs = np.log10([max(p.decisions, lo) for p in pts])
    ys = np.clip([p.score for p in pts], floor, ceil)
    vals = np.interp(grid, xs, ys)
    return float((np.trapezoid(vals, grid) / (grid[-1] - grid[0]) - floor) / (ceil - floor))


def crossover(fast: Sequence[CurvePoint], slow: Sequence[CurvePoint]) -> float | None:
    """First decision count where `slow` (e.g. PPO) reaches or passes `fast` (e.g. Kev). None if it never does."""
    if not fast or not slow:
        return None
    xs = sorted({p.decisions for p in fast} | {p.decisions for p in slow} | {1})
    lx = np.log10(np.maximum(xs, 1))

    def at(curve: Sequence[CurvePoint]) -> np.ndarray:
        pts = sorted(curve, key=lambda p: p.decisions)
        return np.interp(lx, np.log10([max(p.decisions, 1) for p in pts]), [p.score for p in pts])

    diff = at(slow) - at(fast)
    hits = np.flatnonzero(diff >= 0)
    return float(xs[hits[0]]) if len(hits) else None


def efficiency_summary(
    curve: Sequence[CurvePoint], thresholds: Sequence[float], lo: float, hi: float
) -> dict[str, object]:
    return {
        "start": float(sorted(curve, key=lambda p: p.decisions)[0].score) if curve else float("nan"),
        "final": float(sorted(curve, key=lambda p: p.decisions)[-1].score) if curve else float("nan"),
        "to_threshold": {f"{t:g}": decisions_to_threshold(curve, t) for t in thresholds},
        "gain_per_decade": gain_per_decade(curve),
        "aulc": area_under_curve(curve, lo, hi),
    }
