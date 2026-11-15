from __future__ import annotations

import pytest

from fast_and_slow.config import StudyConfig, smoke
from fast_and_slow.notes import templates
from fast_and_slow.notes.bank import NoteBank


@pytest.fixture(scope="session")
def cfg() -> StudyConfig:
    c = smoke()
    c.eval.scenarios_per_horizon = 16
    c.eval.curve_scenarios = 8
    c.classic.total_decisions = 2_000
    return c


@pytest.fixture(scope="session")
def bank(cfg: StudyConfig) -> NoteBank:
    return templates.write(cfg.world, cfg.notes)
