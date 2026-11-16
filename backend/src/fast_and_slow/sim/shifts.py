"""H6 regime changes: what changes in the world, how it shows up, and what a planner writes when declaring it."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import WorldConfig


@dataclass(frozen=True)
class Shift:
    name: str
    label: str
    kind: str  # "announced" (an email says so) | "silent" (only the numbers show it)
    overrides: dict[str, object]
    declared_note: str  # what a planner adds to the brief once the change is declared
    announcements: tuple[str, ...] = field(default=())  # inbox emails on the week it starts (announced shifts)

    def world(self, base: WorldConfig) -> WorldConfig:
        upd = dict(self.overrides)
        if "demand_scale" in upd:
            scale = float(upd.pop("demand_scale"))  # type: ignore[arg-type]
            upd["demand_means"] = tuple(round(m * scale, 3) for m in base.demand_means)
        return base.model_copy(update=upd)


SHIFTS: dict[str, Shift] = {
    "contract_penalty": Shift(
        name="contract_penalty",
        label="A key account adds late-delivery penalties",
        kind="announced",
        overrides={"stockout_penalty": 30.0},
        declared_note=(
            "Change in effect: our largest account's new contract charges $30 for every unit we fail to deliver, "
            "on top of the lost sale. Running short is now far more expensive than carrying extra stock."
        ),
        announcements=(
            "Contract update from Hartwell Foods procurement: effective this week, every unit we fail to deliver is "
            "charged back at $30 on top of the lost sale.",
            "Signed amendment from Marlow Grocers: starting now, short shipments carry a $30 per unit penalty.",
            "Legal sent over the new Northgate Wholesale terms: any unit short on an order is now billed back to us "
            "at $30, effective immediately.",
        ),
    ),
    "competitor_entry": Shift(
        name="competitor_entry",
        label="A competitor enters and demand drops 40%",
        kind="silent",
        overrides={"demand_scale": 0.6},
        declared_note=(
            "Change in effect: a competitor has entered our market. Demand is down about 40% across the board, in "
            "slow weeks, normal weeks and surges alike."
        ),
    ),
}


def get(name: str) -> Shift:
    if name not in SHIFTS:
        raise ValueError(f"unknown shift {name!r}; known: {', '.join(SHIFTS)}")
    return SHIFTS[name]
