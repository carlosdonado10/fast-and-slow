from __future__ import annotations

import logging

import pytest

from fast_and_slow.agents.baselines import RandomAgent
from fast_and_slow.config import StudyConfig
from fast_and_slow.eval.runner import Labeller, eval_scenarios, evaluate
from fast_and_slow.notes.bank import NoteBank
from fast_and_slow.progress import Progress, fmt_duration


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_durations_read_naturally() -> None:
    assert fmt_duration(7) == "7s"
    assert fmt_duration(125) == "2m05s"
    assert fmt_duration(3 * 3600 + 7 * 60) == "3h07m"
    assert fmt_duration(-3) == "0s"


def test_progress_estimates_time_left_from_this_sessions_pace(caplog: pytest.LogCaptureFixture) -> None:
    clock = Clock()
    caplog.set_level(logging.INFO, "fast_and_slow")
    p = Progress("kev_rl H12 seed0", 60, "updates", every=30, done=20, clock=clock)
    assert "resuming at 20" in caplog.records[-1].getMessage()
    assert p.eta() is None
    clock.t = 10
    p.update(21)  # within the throttle window: nothing printed
    assert len(caplog.records) == 1
    clock.t = 40
    p.update(24, "score 61.0%")
    line = caplog.records[-1].getMessage()
    assert "24/60 updates (40%)" in line and "40s in" in line and "about 6m00s left" in line and "score 61.0%" in line
    clock.t = 400
    p.update(60)
    assert "done" in caplog.records[-1].getMessage()
    n = len(caplog.records)
    p.update(60)  # the finish is reported once
    assert len(caplog.records) == n


def test_progress_before_any_work_says_it_is_still_estimating() -> None:
    p = Progress("x", 10, every=0, clock=Clock())
    assert "estimating time left" in p.line()
    p.advance(3, force=True)
    assert p.done == 3


def test_labelled_evaluations_report_progress(
    cfg: StudyConfig, bank: NoteBank, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, "fast_and_slow")
    w = cfg.world
    sc = eval_scenarios(w, 4, 3, 0)
    evaluate(RandomAgent(w), w, bank, sc, Labeller(w, 4, cfg.oracle), record=False, label="refs H4 random")
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs[0].startswith("refs H4 random: starting, 12 decisions")
    assert "12/12 decisions (100%)" in msgs[-1]
