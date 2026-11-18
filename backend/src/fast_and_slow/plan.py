"""What the GPU part of the study will cost, before it costs anything.

`fns kev-check` measures Kev's real speed on the GPU you're on and saves it (speed.json). `fns plan` turns that into
hours and Colab compute units per stage, against your budget, and says how many RL updates fit if it's over.
PPO is not in the bill: it runs on a laptop CPU.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import StudyConfig

# Used only until kev-check has measured the real thing (seconds per request, batched, Kev-4B bf16; rough guesses).
GUESS = {"forward_order": 0.020, "forward_both": 0.035, "train_step": 0.090, "gpu": "guess (run kev-check)"}
FIXED_HOURS = {"setup + model download": 0.15, "note bank (Phi-4-mini)": 0.15}


@dataclass
class Speed:
    forward_order: float  # s per request, order question only (RL rollouts and curve evals)
    forward_both: float  # s per request, order + regime questions (final evals)
    train_step: float  # s per sample for forward + backward in a PPO minibatch
    gpu: str

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> Speed:
        return cls(**json.loads(path.read_text())) if path.exists() else cls(**GUESS)  # type: ignore[arg-type]


@dataclass
class Line:
    stage: str
    decisions: int
    hours: float


def rl_hours(cfg: StudyConfig, speed: Speed, h: int, updates: int | None = None) -> tuple[int, float]:
    k, e = cfg.kev, cfg.eval
    updates = k.rl_updates if updates is None else updates
    decisions = updates * k.rl_groups_per_update * k.rl_group_size * h
    rollout = decisions * speed.forward_order
    learn = decisions * k.rl_epochs * speed.train_step
    curve_evals = (updates // max(1, k.eval_every_updates) + 2) * e.curve_scenarios * h * speed.forward_order
    final = e.scenarios_per_horizon * h * speed.forward_both
    return decisions, (rollout + learn + curve_evals + final) / 3600


def lines(cfg: StudyConfig, speed: Speed) -> list[Line]:
    e = cfg.eval
    out = [Line(name, 0, hrs) for name, hrs in FIXED_HOURS.items()]
    for arm in ("kev_zero", "kev_icl"):
        n = sum(e.scenarios_per_horizon * h for h in e.horizons)
        out.append(Line(f"{arm} (H={','.join(map(str, e.horizons))})", n, n * speed.forward_both / 3600))
    for s in cfg.seeds:
        for h in cfg.kev.rl_horizons:
            d, hrs = rl_hours(cfg, speed, h)
            out.append(Line(f"kev_rl H={h} seed {s}", d, hrs))
    sh = cfg.shift
    per_shift = 2 * sh.scenarios * sh.horizon + (sh.detect_streams + sh.calibration_streams) * sh.weeks
    n = per_shift * len(sh.shifts)
    out.append(Line(f"H6 shifts (Kev + alarm, {len(sh.shifts)})", n, n * speed.forward_order / 3600))
    return out


def updates_that_fit(cfg: StudyConfig, speed: Speed, budget_hours: float) -> int:
    """Largest rl_updates (same for every RL run) that keeps the whole GPU bill within budget_hours."""
    fixed = sum(ln.hours for ln in lines(cfg, speed) if not ln.stage.startswith("kev_rl"))
    runs = [(s, h) for s in cfg.seeds for h in cfg.kev.rl_horizons]
    lo, hi = 0, 10_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        total = fixed + sum(rl_hours(cfg, speed, h, mid)[1] for _, h in runs)
        lo, hi = (mid, hi) if total <= budget_hours else (lo, mid - 1)
    return lo


def sessions_needed(work_hours: float, session_hours: float, setup_hours: float) -> int | None:
    """Colab sessions to get through work_hours when each one first spends setup_hours reloading. None if none fit."""
    useful = session_hours - setup_hours
    return math.ceil(work_hours / useful) if useful > 0 else None


def render(
    cfg: StudyConfig,
    speed: Speed,
    cu_per_hour: float,
    budget_cu: float,
    margin: float = 0.8,
    session_cu: float = 0.0,
) -> str:
    rows = lines(cfg, speed)
    total_h = sum(r.hours for r in rows)
    out = [f"GPU speed: {speed.gpu}  |  {cu_per_hour:g} compute units per hour  |  budget {budget_cu:g} CU", ""]
    out.append(f"{'stage':34s} {'decisions':>10s} {'hours':>7s} {'CU':>7s}")
    for r in rows:
        out.append(f"{r.stage:34s} {r.decisions:>10,d} {r.hours:>7.2f} {r.hours * cu_per_hour:>7.1f}")
    total_cu = total_h * cu_per_hour
    out.append(f"{'total (GPU)':34s} {'':>10s} {total_h:>7.2f} {total_cu:>7.1f}")
    out.append("PPO: runs on a laptop CPU, 0 CU.")
    usable = budget_cu * margin
    if total_cu <= usable:
        out.append(
            f"Fits: {total_cu:.0f} of {budget_cu:g} CU, keeping {100 * (1 - margin):.0f}% in reserve for reruns."
        )
    else:
        fit = updates_that_fit(cfg, speed, usable / cu_per_hour)
        out.append(
            f"Over budget: {total_cu:.0f} CU planned, {usable:.0f} usable ({100 * margin:.0f}% of {budget_cu:g})."
        )
        if fit > 0:
            out.append(f"Set kev.rl_updates to {fit} (now {cfg.kev.rl_updates}) or drop an RL horizon or seed.")
        else:
            out.append("Even 1 RL update doesn't fit: drop RL horizons or seeds, or use more units.")
    if session_cu > 0:
        setup = FIXED_HOURS["setup + model download"]
        session_h = session_cu / cu_per_hour
        n = sessions_needed(total_h - setup, session_h, setup)
        if n is None:
            out.append(f"Sessions of {session_cu:g} CU are too short: setup alone takes {setup * cu_per_hour:.1f} CU.")
        else:
            cap = int((session_h - setup) * 60)
            out.append(
                f"In sessions of {session_cu:g} CU: about {n}, each paying {setup * cu_per_hour:.1f} CU of setup "
                f"({(total_h + (n - 1) * setup) * cu_per_hour:.0f} CU in all). Set MAX_MINUTES to {cap}."
            )
    if math.isclose(speed.forward_order, GUESS["forward_order"]) and speed.gpu == GUESS["gpu"]:  # type: ignore[arg-type]
        out.append("These are guesses. Run `fns kev-check` on the GPU first for measured numbers.")
    return "\n".join(out)
