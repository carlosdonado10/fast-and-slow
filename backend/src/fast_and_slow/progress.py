"""Progress lines for long stages, so a Colab cell says how far along it is and roughly how long is left.

One plain line at a time (no carriage-return bars): the notebook streams output line by line, and the lines stay
readable in a log file. A line is printed at most every `every` seconds, plus once at the start and once at the end.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

log = logging.getLogger("fast_and_slow")


def fmt_duration(seconds: float) -> str:
    s = max(0, round(seconds))
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


class Progress:
    """`done` may start above zero (a resumed run); the time-left estimate uses only this session's pace."""

    def __init__(
        self,
        label: str,
        total: int,
        unit: str = "",
        every: float = 30.0,
        done: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.label, self.total, self.unit, self.every, self.clock = label, max(1, total), unit, every, clock
        self.start_done = self.done = done
        self.t0 = self.last = clock()
        self.finished = False
        resumed = f", resuming at {done:,}" if done else ""
        log.info("%s: starting, %s %s%s", label, f"{self.total:,}", unit, resumed)

    def eta(self) -> float | None:
        gained = self.done - self.start_done
        if gained <= 0 or self.done >= self.total:
            return None
        return (self.clock() - self.t0) / gained * (self.total - self.done)

    def line(self, note: str = "") -> str:
        pct = 100 * self.done / self.total
        unit = f" {self.unit}" if self.unit else ""
        parts = [f"{self.label}: {self.done:,}/{self.total:,}{unit} ({pct:.0f}%)", f"{fmt_duration(self.elapsed)} in"]
        if self.done >= self.total:
            parts.append("done")
        else:
            eta = self.eta()
            parts.append(f"about {fmt_duration(eta)} left" if eta is not None else "estimating time left")
        if note:
            parts.append(note)
        return ", ".join(parts)

    @property
    def elapsed(self) -> float:
        return self.clock() - self.t0

    def update(self, done: int, note: str = "", force: bool = False) -> None:
        self.done = min(done, self.total)
        now = self.clock()
        complete = self.done >= self.total
        if complete and self.finished:
            return
        if not (force or complete or now - self.last >= self.every):
            return
        self.last = now
        self.finished = complete
        log.info(self.line(note))

    def advance(self, n: int = 1, note: str = "", force: bool = False) -> None:
        self.update(self.done + n, note, force)
