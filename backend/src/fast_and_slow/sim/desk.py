"""An episode on the desk: the shared stepping logic every agent and every evaluator goes through."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import WorldConfig
from .world import Note, PeriodOutcome, Scenario, step_economics


@dataclass(frozen=True)
class SeenNote:
    week: int
    note: Note
    text: str | None  # None for "no note this week"


@dataclass
class Observation:
    """What a planner sees at the start of a week, before ordering. Never contains the hidden regime."""

    week: int  # 0-based
    horizon: int
    inventory: int
    history: list[PeriodOutcome]  # past weeks, oldest first
    notes: list[SeenNote]  # every note so far including this week's, oldest first

    @property
    def this_week_note(self) -> SeenNote:
        return self.notes[-1]

    @property
    def weeks_left(self) -> int:
        return self.horizon - self.week


@dataclass
class Episode:
    world: WorldConfig
    scenario: Scenario
    note_texts: list[str | None]  # resolved paraphrase for each week's note
    week: int = 0
    inventory: int = 0
    history: list[PeriodOutcome] = field(default_factory=list)
    total: float = 0.0

    def __post_init__(self) -> None:
        self.inventory = self.world.start_inventory

    @property
    def done(self) -> bool:
        return self.week >= self.scenario.horizon

    def observe(self) -> Observation:
        seen = [SeenNote(t, self.scenario.notes[t], self.note_texts[t]) for t in range(self.week + 1)]
        return Observation(self.week, self.scenario.horizon, self.inventory, list(self.history), seen)

    def step(self, order: int) -> PeriodOutcome:
        if self.done:
            raise RuntimeError("episode is over")
        t = self.week
        out = step_economics(
            self.world, self.inventory, order, self.scenario.demands[t], last=t == self.scenario.horizon - 1
        )
        self.history.append(out)
        self.inventory = out.inventory_end
        self.total += out.reward
        self.week += 1
        return out
