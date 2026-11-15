from __future__ import annotations

import math

import pytest

from fast_and_slow.eval import metrics as M
from fast_and_slow.eval.runner import DecisionRecord


def rec(conf: float, ok: bool, decidable: bool = True, regime: list[float] | None = None) -> DecisionRecord:
    return DecisionRecord(
        seed=0,
        week=0,
        inventory=0,
        action=0,
        confidence=conf,
        acceptable=ok,
        decidable=decidable,
        bayes_action=0,
        regret=0.0 if ok else 5.0,
        true_regime=1,
        bayes_posterior=[0.1, 0.8, 0.1],
        regime_probs=regime,
        latency_ms=1.0,
        reward=1.0,
    )


def test_score_scales() -> None:
    assert M.pct_of_optimal(50, 0, 100) == 50
    assert M.pct_of_optimal(-10, 0, 100) == -10
    assert M.inbox_value(80, 60, 100) == 50


def test_bootstrap_ci_contains_the_mean_and_handles_tiny_inputs() -> None:
    lo, hi = M.bootstrap_ci([1, 2, 3, 4, 5], n=500)
    assert lo <= 3 <= hi
    assert M.bootstrap_ci([7]) == (7, 7)
    assert all(math.isnan(x) for x in M.bootstrap_ci([]))


def test_reliability_of_a_perfectly_calibrated_agent() -> None:
    confs = [0.9] * 10 + [0.5] * 10
    oks = [True] * 9 + [False] + [True] * 5 + [False] * 5
    r = M.reliability(confs, oks)
    assert r.ece == pytest.approx(0, abs=1e-9)
    assert sum(b["count"] for b in r.bins) == 20
    assert M.reliability([1.0], [True]).bins[-1]["hi"] == 1.0


def test_regime_calibration_compares_to_truth_and_to_bayes() -> None:
    assert M.regime_calibration([rec(0.5, True)]) is None
    out = M.regime_calibration([rec(0.5, True, regime=[0.1, 0.8, 0.1])])
    assert out is not None
    assert out["brier_vs_bayes"] == pytest.approx(0)
    assert out["brier_vs_truth"] == pytest.approx(0.01 + 0.04 + 0.01)


def test_automation_takes_the_confident_correct_prefix_within_budget() -> None:
    confs = [0.99, 0.95, 0.9, 0.6, 0.5]
    oks = [True, True, True, False, True]
    assert M.automation_at_budget(confs, oks, 0.0) == pytest.approx(0.6)
    assert M.automation_at_budget(confs, oks, 0.25) == pytest.approx(1.0)
    assert M.automation_at_budget([0.9], [False], 0.05) == 0.0
    assert M.automation_at_budget([], [], 0.05) == 0.0


def test_handoff_splits_decidable_and_judgement() -> None:
    rs = [rec(0.9, True, True), rec(0.8, True, True), rec(0.7, False, False), rec(0.6, True, False)]
    out = M.handoff(rs, (0.05,))
    assert out["decidable"]["automation"]["0.05"] == 1.0  # type: ignore[index]
    assert out["judgement"]["automation"]["0.05"] == 0.0  # type: ignore[index]
    assert out["all"]["n"] == 4  # type: ignore[index]


CURVE = [M.CurvePoint(0, 40), M.CurvePoint(100, 60), M.CurvePoint(1_000, 80), M.CurvePoint(10_000, 90)]


def test_decisions_to_threshold_interpolates_in_log_space() -> None:
    assert M.decisions_to_threshold(CURVE, 30) == 0
    assert M.decisions_to_threshold(CURVE, 80) == pytest.approx(1_000)
    assert M.decisions_to_threshold(CURVE, 70) == pytest.approx(10**2.5)
    assert M.decisions_to_threshold(CURVE, 95) == math.inf
    assert M.decisions_to_threshold([], 50) == math.inf


def test_gain_per_decade_and_area() -> None:
    assert M.gain_per_decade(CURVE) == pytest.approx(15, abs=1e-6)
    assert math.isnan(M.gain_per_decade(CURVE[:2]))
    assert math.isnan(M.gain_per_decade([M.CurvePoint(5, 1), M.CurvePoint(5, 2)]))
    flat = [M.CurvePoint(0, 50)]
    assert M.area_under_curve(flat, 10, 10_000) == pytest.approx(0.5)
    assert M.area_under_curve(CURVE, 10, 10_000) > M.area_under_curve(
        [M.CurvePoint(0, 0), M.CurvePoint(10_000, 90)], 10, 10_000
    )
    assert math.isnan(M.area_under_curve([], 10, 100))


def test_crossover_finds_where_the_slow_learner_catches_up() -> None:
    fast = [M.CurvePoint(0, 70), M.CurvePoint(10_000, 75)]
    slow = [M.CurvePoint(0, 0), M.CurvePoint(100, 50), M.CurvePoint(1_000, 80), M.CurvePoint(10_000, 95)]
    assert M.crossover(fast, slow) == 1_000
    assert M.crossover(fast, [M.CurvePoint(0, 0), M.CurvePoint(10_000, 10)]) is None
    assert M.crossover([], slow) is None


def test_efficiency_summary_shape() -> None:
    s = M.efficiency_summary(CURVE, (80, 90), 10, 10_000)
    assert s["start"] == 40 and s["final"] == 90
    assert set(s["to_threshold"]) == {"80", "90"}  # type: ignore[arg-type]
    empty = M.efficiency_summary([], (80,), 10, 100)
    assert math.isnan(empty["start"])  # type: ignore[arg-type]
