"""H6: the calendar math behind the transition timelines, and the two change detectors."""

from __future__ import annotations

import numpy as np
import pytest

from fast_and_slow.agents.baselines import BaseStockAgent
from fast_and_slow.config import ShiftConfig, StudyConfig
from fast_and_slow.eval import detect as D
from fast_and_slow.eval.transition import CurvePt, bridge_saving, decisions_available, score_at, timeline
from fast_and_slow.notes.bank import NoteBank
from fast_and_slow.sim.shifts import SHIFTS, get

CFG = ShiftConfig(weeks=20, shift_week=8, rebuild_weeks=4, live_units=10)
CURVE = [CurvePt(0, 40.0), CurvePt(100, 70.0), CurvePt(1_000, 95.0)]


def test_score_at_is_a_step_function() -> None:
    assert score_at(CURVE, 0) == 40 and score_at(CURVE, 99) == 40 and score_at(CURVE, 100) == 70
    assert score_at(CURVE, 10**9) == 95


def test_experience_arrives_differently_per_mode() -> None:
    assert decisions_available(CFG, 9, 8, "sim", 1_000) == 0  # simulator still being rebuilt
    assert decisions_available(CFG, 12, 8, "sim", 1_000) == 1_000
    assert decisions_available(CFG, 12, 8, "live", 1_000) == 40  # 10 units x 4 weeks
    assert decisions_available(CFG, 5, 8, "live", 1_000) == 0
    with pytest.raises(ValueError):
        decisions_available(CFG, 9, 8, "telepathy", 1)


def test_bridge_covers_until_retrained_rl_wins_then_hands_back() -> None:
    t = timeline(CFG, pre_score=90, curve=CURVE, kev_score=80, trigger=8, mode="sim", usd_gap_per_week=100)
    assert t.weekly["stale"][:8] == [90] * 8 and t.weekly["stale"][8:] == [40] * 12
    assert t.weekly["bridge"][8:12] == [80] * 4  # Kev covers while the simulator is rebuilt
    assert t.weekly["bridge"][12:] == [95] * 8 and t.handback_week == 4
    assert t.regret_usd["bridge"] < t.regret_usd["retrain"] < t.regret_usd["stale"]
    assert t.recover_week["retrain"] == 4 and t.recover_week["stale"] is None
    assert 0 < bridge_saving(t) < 1


def test_bridge_never_returns_to_the_stale_policy_and_nobody_noticing_means_stale() -> None:
    t = timeline(CFG, 90, CURVE, kev_score=30, trigger=8, mode="live", usd_gap_per_week=100)
    assert t.weekly["bridge"][8] == 30  # Kev is bad here, but the declared-stale policy stays benched
    unnoticed = timeline(CFG, 90, CURVE, kev_score=80, trigger=None, mode="sim", usd_gap_per_week=100)
    assert unnoticed.weekly["bridge"] == unnoticed.weekly["stale"] and unnoticed.handback_week is None
    perfect = timeline(CFG, 90, [CurvePt(0, 100.0)], 100, 8, "sim", 100)
    assert bridge_saving(perfect) == 0.0


def test_shift_worlds() -> None:
    w = StudyConfig().world
    assert get("contract_penalty").world(w).stockout_penalty == 30
    assert get("competitor_entry").world(w).demand_means == pytest.approx(tuple(0.6 * m for m in w.demand_means))
    assert SHIFTS["contract_penalty"].kind == "announced" and SHIFTS["competitor_entry"].announcements == ()
    with pytest.raises(ValueError):
        get("asteroid")


def test_streams_put_the_announcement_on_the_shift_week(cfg: StudyConfig, bank: NoteBank) -> None:
    sh = get("contract_penalty")
    old, new = cfg.world, sh.world(cfg.world)
    tool = BaseStockAgent(old, 10, 20)
    st = D.make_stream(old, new, sh, bank, 12, 6, seed=1, tool=tool)
    assert len(st.observations) == 12 and st.observations[6].this_week_note.text in sh.announcements
    assert all(o.this_week_note.text not in sh.announcements for o in st.observations[:6])
    calm = D.make_stream(old, old, None, bank, 12, None, seed=1, tool=tool)
    assert calm.shift_week is None


def test_page_hinkley_fires_on_a_silent_demand_drop_and_calibration_matches_the_target(
    cfg: StudyConfig, bank: NoteBank
) -> None:
    sh = get("competitor_entry")
    old, new = cfg.world, sh.world(cfg.world)
    tool = BaseStockAgent(old, 10, 20)
    calm = [D.page_hinkley(old, D.make_stream(old, old, None, bank, 40, None, 100 + i, tool)) for i in range(20)]
    thr = D.calibrate(calm, 0.05)
    flat = np.concatenate([np.asarray(s[4:]) for s in calm])
    assert np.mean(flat > thr) == pytest.approx(0.05, abs=0.02)
    shifted = [D.make_stream(old, new, sh, bank, 40, 20, 200 + i, tool) for i in range(10)]
    summary = D.summarise([D.page_hinkley(old, s) for s in shifted], shifted, thr)
    assert summary["detection_rate"] >= 0.7  # type: ignore[operator]
    assert summary["median_delay_weeks"] is not None
    none = D.summarise([[0.0] * 40], shifted[:1], thr)
    assert none["detection_rate"] == 0 and none["median_delay_weeks"] is None
