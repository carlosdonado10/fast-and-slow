"""H6 detection: would anyone notice the world changed? Kev's weekly alarm vs a Page-Hinkley test on demand.

A stream is one product's year: `weeks` weeks of the old world, switching to the new one at `shift_week`. The
inventory follows the numbers-only base-stock tool (so the history table looks like a real desk). Announced shifts put
an email in the inbox on the week they start; silent ones don't.

Both detectors produce a score per week; each one's threshold is set on separate no-shift streams so that it raises
the same rate of false alarms (`false_alarm_target` per week). Then delay and detection rate are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..agents.baselines import BaseStockAgent
from ..config import WorldConfig
from ..notes.bank import NoteBank
from ..oracle.bayes import BayesPlanner
from ..sim.desk import Observation, SeenNote
from ..sim.shifts import Shift
from ..sim.world import DISTRACTOR, Note, PeriodOutcome, demand_pmf, draw_note, step_economics, transition_matrix


@dataclass
class Stream:
    seed: int
    shift_week: int | None  # None: a calibration stream that never shifts
    observations: list[Observation]
    demands: list[int]


def make_stream(
    old: WorldConfig,
    new: WorldConfig,
    shift: Shift | None,
    bank: NoteBank,
    weeks: int,
    shift_week: int | None,
    seed: int,
    tool: BaseStockAgent,
) -> Stream:
    rng = np.random.default_rng(seed)
    k = len(old.regimes)
    z = int(rng.choice(k, p=np.asarray(old.regime_prior)))
    inv = old.start_inventory
    history: list[PeriodOutcome] = []
    seen: list[SeenNote] = []
    obs: list[Observation] = []
    demands: list[int] = []
    for t in range(weeks):
        w = new if shift_week is not None and t >= shift_week else old
        if t > 0:
            z = int(rng.choice(k, p=transition_matrix(w)[z]))
        d = int(rng.choice(w.max_demand + 1, p=demand_pmf(w)[z]))
        note = draw_note(w, rng, z)
        text = bank.text(note, "test")
        if shift is not None and shift.announcements and t == shift_week:
            note, text = Note(DISTRACTOR), shift.announcements[seed % len(shift.announcements)]
        seen.append(SeenNote(t, note, text))
        o = Observation(t, weeks, inv, list(history), list(seen))
        obs.append(o)
        order = w.order_menu[tool.act(o, 0).action]
        out = step_economics(w, inv, order, d, last=False)
        history.append(out)
        inv = out.inventory_end
        demands.append(d)
    return Stream(seed, shift_week, obs, demands)


def page_hinkley(old: WorldConfig, stream: Stream, drift: float = 0.25) -> list[float]:
    """Two-sided CUSUM on standardised demand surprises under the *old* world's belief (notes read perfectly)."""
    planner = BayesPlanner(old, 1)
    means = np.asarray(old.demand_means)
    b = planner.prior()
    up = down = 0.0
    scores = []
    for o, d in zip(stream.observations, stream.demands, strict=True):
        b = planner.after_note(b, o.this_week_note.note)
        m = float(b @ means)
        v = float(b @ (means + means**2)) - m**2
        zt = (d - m) / np.sqrt(max(v, 1e-6))
        up, down = max(0.0, up + zt - drift), max(0.0, down - zt - drift)
        scores.append(max(up, down))
        b = planner.after_demand(b, d)
    return scores


def calibrate(no_shift_scores: list[list[float]], target: float, burn_in: int = 4) -> float:
    """Threshold whose per-week exceedance rate on no-shift streams is `target`."""
    flat = np.concatenate([np.asarray(s[burn_in:]) for s in no_shift_scores])
    return float(np.quantile(flat, 1 - target))


def summarise(
    scores: list[list[float]], streams: list[Stream], threshold: float, burn_in: int = 4
) -> dict[str, object]:
    delays: list[int | None] = []
    pre_alarms = pre_weeks = 0
    for sc, st in zip(scores, streams, strict=True):
        assert st.shift_week is not None
        pre = sc[burn_in : st.shift_week]
        pre_alarms += sum(x > threshold for x in pre)
        pre_weeks += len(pre)
        hit = next((t for t in range(st.shift_week, len(sc)) if sc[t] > threshold), None)
        delays.append(None if hit is None else hit - st.shift_week)
    found = [d for d in delays if d is not None]
    return {
        "threshold": threshold,
        "false_alarms_per_100_weeks": 100 * pre_alarms / max(1, pre_weeks),
        "detection_rate": len(found) / max(1, len(delays)),
        "median_delay_weeks": float(np.median(found)) if found else None,
        "delays": delays,
    }
