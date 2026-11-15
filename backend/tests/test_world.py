from __future__ import annotations

import numpy as np
import pytest

from fast_and_slow.config import WorldConfig, load, smoke
from fast_and_slow.sim.desk import Episode
from fast_and_slow.sim.world import (
    DISTRACTOR,
    NONE,
    SIGNAL,
    Note,
    demand_pmf,
    note_likelihood,
    note_outcomes,
    sample_scenario,
    step_economics,
    transition_matrix,
)


def test_transition_rows_sum_to_one() -> None:
    T = transition_matrix(WorldConfig())
    assert np.allclose(T.sum(1), 1)
    assert np.allclose(np.diag(T), WorldConfig().stay_prob)


def test_demand_pmf_is_a_distribution_with_the_right_means() -> None:
    w = WorldConfig()
    pmf = demand_pmf(w)
    assert pmf.shape == (3, w.max_demand + 1)
    assert np.allclose(pmf.sum(1), 1)
    means = pmf @ np.arange(w.max_demand + 1)
    assert np.allclose(means, w.demand_means, atol=0.05)


def test_note_outcomes_cover_all_probability_mass_under_every_regime() -> None:
    w = WorldConfig()
    total = sum(p for _, p in note_outcomes(w))
    assert np.allclose(total, 1.0)


def test_signal_likelihood_favours_the_claimed_regime_by_source_accuracy() -> None:
    w = WorldConfig()
    lik = note_likelihood(w, Note(SIGNAL, source=2, claim=0))
    assert lik[0] == pytest.approx(w.source_accuracy[2])
    assert lik[1] == pytest.approx((1 - w.source_accuracy[2]) / 2)
    assert np.allclose(note_likelihood(w, Note(DISTRACTOR)), 1)
    assert Note(SIGNAL, 1, 2).cell == "1:2" and Note(NONE).cell == NONE


def test_scenarios_are_deterministic_per_seed_and_differ_across_seeds() -> None:
    w = WorldConfig()
    a, b, c = sample_scenario(w, 12, 5), sample_scenario(w, 12, 5), sample_scenario(w, 12, 6)
    assert a == b
    assert a != c
    assert len(a.demands) == len(a.notes) == len(a.regimes) == 12


def test_economics_of_one_period() -> None:
    w = WorldConfig()
    out = step_economics(w, inventory=5, order=10, demand=12, last=False)
    assert (out.order, out.sales, out.lost, out.inventory_end) == (10, 12, 0, 3)
    expected = w.price * 12 - w.unit_cost * 10 - w.fixed_order_cost - w.holding_cost * 3
    assert out.reward == pytest.approx(expected)
    short = step_economics(w, inventory=0, order=0, demand=4, last=True)
    assert short.lost == 4 and short.reward == pytest.approx(-w.stockout_penalty * 4)


def test_orders_are_clipped_to_the_shelf() -> None:
    w = WorldConfig()
    out = step_economics(w, inventory=w.inventory_cap - 5, order=40, demand=0, last=False)
    assert out.order == 5


def test_episode_steps_and_ends() -> None:
    w = WorldConfig()
    sc = sample_scenario(w, 3, 1)
    ep = Episode(w, sc, ["a", None, "c"])
    obs = ep.observe()
    assert obs.week == 0 and obs.weeks_left == 3 and obs.this_week_note.text == "a"
    for _ in range(3):
        ep.step(10)
    assert ep.done
    with pytest.raises(RuntimeError):
        ep.step(10)


def test_profiles_and_yaml_overrides(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert load("smoke").kev.fake is True
    assert load("full").notes.writer == "llm"
    f = tmp_path / "c.yaml"
    f.write_text("profile: smoke\neval:\n  horizons: [2]\nkev:\n  rl_updates: 1\n")
    c = load(str(f))
    assert c.eval.horizons == (2,) and c.kev.rl_updates == 1 and c.kev.fake is True
    assert smoke().profile == "smoke"
