from __future__ import annotations

import numpy as np
import pytest

from fast_and_slow.agents.baselines import BaseStockAgent, BayesAgent, OracleAgent, RandomAgent, bayes_belief
from fast_and_slow.config import StudyConfig
from fast_and_slow.eval.runner import EvalResult, Labeller, eval_scenarios, evaluate
from fast_and_slow.notes.bank import NoteBank
from fast_and_slow.oracle.bayes import BayesPlanner, PerfectInfoPlanner, simplex_grid
from fast_and_slow.sim.desk import Episode
from fast_and_slow.sim.world import SIGNAL, Note, sample_scenario


def test_simplex_grid_and_nearest_round_trip() -> None:
    bp = BayesPlanner(StudyConfig().world, 1)
    grid = simplex_grid(3, 20)
    assert len(grid) == 231 and np.allclose(grid.sum(1), 1)
    idx = bp.tab.nearest(grid)
    assert (idx == np.arange(len(grid))).all()
    assert int(bp.tab.nearest(np.array([0.333, 0.333, 0.334]))) in range(len(grid))
    with pytest.raises(NotImplementedError):
        simplex_grid(4, 5)


def test_a_reliable_note_moves_the_belief_towards_its_claim() -> None:
    bp = BayesPlanner(StudyConfig().world, 2)
    b = bp.after_note(bp.prior(), Note(SIGNAL, source=2, claim=2))
    assert b.argmax() == 2 and b[2] > 0.8
    b2 = bp.after_demand(np.array([0, 0, 1.0]), demand=24)
    assert np.isclose(b2.sum(), 1)


def test_bayes_beats_basestock_beats_random_and_oracle_bounds_bayes(cfg: StudyConfig, bank: NoteBank) -> None:
    w, h = cfg.world, 4
    lab = Labeller(w, h, cfg.oracle)
    sc = eval_scenarios(w, h, 60, 10_000)
    mean = {
        a.name: np.mean(evaluate(a, w, bank, sc, lab, record=False).returns)
        for a in [RandomAgent(w), BaseStockAgent.tune(w, h, list(range(40))), BayesAgent(w, h), OracleAgent(w, h)]
    }
    assert mean["oracle"] > mean["bayes"] > mean["basestock"] > mean["random"]


def test_q_values_pick_more_stock_when_a_surge_is_certain() -> None:
    w = StudyConfig().world
    bp = BayesPlanner(w, 1)
    low = w.order_menu[int(np.argmax(bp.q_values(0, 0, np.array([1.0, 0, 0]))))]
    surge = w.order_menu[int(np.argmax(bp.q_values(0, 0, np.array([0, 0, 1.0]))))]
    assert surge > low
    q = bp.q_values(0, 0, np.array([0, 1.0, 0]))
    assert bp.acceptable(q)[int(np.argmax(q))]


def test_perfect_info_planner_values_dominate_and_are_consistent() -> None:
    w = StudyConfig().world
    pi = PerfectInfoPlanner(w, 3)
    assert pi.Q[0].shape == (3, w.inventory_cap + 1, len(w.order_menu))
    assert pi.best_action(0, 0, 2) >= pi.best_action(0, 0, 0)
    assert np.allclose(pi.q_values(1, 0, 1), pi.Q[1][1, 0])


def test_evaluate_labels_every_decision_and_round_trips(cfg: StudyConfig, bank: NoteBank) -> None:
    w, h = cfg.world, 3
    lab = Labeller(w, h, cfg.oracle)
    sc = eval_scenarios(w, h, 5, 10_000)
    r = evaluate(BayesAgent(w, h), w, bank, sc, lab)
    assert len(r.decisions) == 5 * h
    assert all(d.acceptable and d.regret == pytest.approx(0, abs=1e-6) for d in r.decisions)
    assert EvalResult.from_json(r.to_json()).returns == r.returns


def test_bayes_belief_replays_history(cfg: StudyConfig, bank: NoteBank) -> None:
    w = cfg.world
    agent = BayesAgent(w, 3)
    sc = sample_scenario(w, 3, 3)
    ep = Episode(w, sc, bank.texts(sc, "test"))
    ep.step(10)
    b = bayes_belief(agent.planner, ep.observe())
    assert np.isclose(b.sum(), 1) and b.shape == (3,)
