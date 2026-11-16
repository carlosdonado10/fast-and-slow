"""H6: what each strategy earns, week by week, from the moment the world changes (pure functions, no I/O).

Inputs are measured elsewhere: the stale and retraining PPO scores on the new world (one warm-started retraining
curve), Kev's score on the new world with the updated brief, and when the change was declared or detected. These
functions only map them onto the calendar, so the same measurements serve both retraining modes and both triggers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..config import ShiftConfig

ARMS = ("stale", "retrain", "kev", "bridge")


@dataclass(frozen=True)
class CurvePt:
    decisions: int
    score: float


def score_at(curve: Sequence[CurvePt], decisions: float) -> float:
    """Score of the latest retraining checkpoint at or before `decisions` (a step function, no interpolation)."""
    pts = sorted(curve, key=lambda p: p.decisions)
    best = pts[0].score
    for p in pts:
        if p.decisions <= decisions:
            best = p.score
    return best


def decisions_available(cfg: ShiftConfig, week: int, trigger: int, mode: str, full: int) -> float:
    """New-world experience the retrained policy has by `week`, if retraining started at `trigger`."""
    if week < trigger:
        return 0.0
    if mode == "sim":  # rebuild the simulator first; offline retraining is then effectively instant on the calendar
        return float(full) if week >= trigger + cfg.rebuild_weeks else 0.0
    if mode == "live":  # learn only from weeks actually played, pooled over live_units products
        return float(cfg.live_units * (week - trigger))
    raise ValueError(f"unknown retraining mode {mode!r}")


@dataclass
class Timeline:
    weekly: dict[str, list[float]]  # arm -> % of optimal per week (old world's optimum before the shift)
    regret_usd: dict[str, float]  # arm -> profit lost vs Bayes, shift to end
    recover_week: dict[str, int | None]  # arm -> weeks after the shift until >= recover_at (None: never)
    handback_week: int | None  # weeks after the shift when the bridge hands back to RL


def timeline(
    cfg: ShiftConfig,
    pre_score: float,
    curve: Sequence[CurvePt],
    kev_score: float,
    trigger: int | None,
    mode: str,
    usd_gap_per_week: float,
) -> Timeline:
    """trigger: the week the change is declared or detected (None: nobody ever notices)."""
    full = max(p.decisions for p in curve)
    stale = score_at(curve, 0)
    weekly: dict[str, list[float]] = {a: [] for a in ARMS}
    handback: int | None = None
    for w in range(cfg.weeks):
        if w < cfg.shift_week:
            for a in ARMS:
                weekly[a].append(pre_score)
            continue
        noticed = trigger is not None and w >= trigger
        experience = decisions_available(cfg, w, trigger, mode, full) if noticed else 0.0  # type: ignore[arg-type]
        retrained = score_at(curve, experience) if noticed else stale
        weekly["stale"].append(stale)
        weekly["retrain"].append(retrained)
        weekly["kev"].append(kev_score if noticed else stale)
        if not noticed:
            weekly["bridge"].append(stale)
        elif handback is not None or (experience > 0 and retrained > kev_score):
            # once a change is declared the old policy is benched; RL returns only as a *retrained* policy that
            # beats the bridge (nobody can score the stale policy on the new world before retraining data exists)
            handback = w - cfg.shift_week if handback is None else handback
            weekly["bridge"].append(retrained)
        else:
            weekly["bridge"].append(kev_score)
    post = slice(cfg.shift_week, cfg.weeks)
    regret = {a: sum((100 - s) / 100 * usd_gap_per_week for s in weekly[a][post]) for a in ARMS}
    recover = {a: next((i for i, s in enumerate(weekly[a][post]) if s >= cfg.recover_at), None) for a in ARMS}
    return Timeline(weekly, regret, recover, handback)


def bridge_saving(t: Timeline) -> float:
    """Share of transition regret the bridge removes vs the better RL-only strategy (stale or retrain)."""
    best = min(t.regret_usd["stale"], t.regret_usd["retrain"])
    return 0.0 if best <= 0 else (best - t.regret_usd["bridge"]) / best
