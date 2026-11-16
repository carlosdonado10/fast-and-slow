"""Turns an Observation into what a text-reading agent sees: a System One request (state + typed questions).

Kept free of any Kev import: the request is plain JSON in the TypeSafe / Kev API shape, so it can be logged, shown in the
report's replays and sent to a local model or a server alike.
"""

from __future__ import annotations

from typing import Any

from ..config import WorldConfig
from .desk import Observation

ORDER_Q = "order"
REGIME_Q = "regime"
CHANGE_Q = "changed"


def order_key(q: int) -> str:
    return f"order_{q}"


def business_brief(w: WorldConfig) -> str:
    return (
        f"You are the replenishment planner for one product. Each week you place one order before customers buy. "
        f"Stock sells at ${w.price:.0f} a unit and costs ${w.unit_cost:.0f}; every order also costs a fixed "
        f"${w.fixed_order_cost:.0f}. Unsold stock carries over and costs ${w.holding_cost:.0f} a unit per week. "
        f"Each unit of demand you can't meet costs ${w.stockout_penalty:.0f} in lost goodwill on top of the lost sale. "
        f"At the end of the plan leftovers are cleared at ${w.salvage_value:.0f} a unit. Usual weekly demand is about "
        f"{w.demand_means[1]:.0f} units, lower in a slow spell and much higher in a surge. Notes in your inbox vary in "
        f"reliability: rumours are often wrong, sales reps are usually right, written customer confirmations rarely wrong."
    )


def state_text(w: WorldConfig, obs: Observation, context: str | None = None) -> str:
    lines = [business_brief(w), ""]
    if context:
        lines += [context, ""]
    lines.append(f"Week {obs.week + 1} of {obs.horizon} ({obs.weeks_left} including this one left in the plan).")
    lines.append(f"Stock on hand now: {obs.inventory} units.")
    if obs.history:
        lines.append("Past weeks (ordered / demand / sold / missed):")
        for t, h in enumerate(obs.history[-6:], start=max(0, len(obs.history) - 6)):
            lines.append(f"- Week {t + 1}: {h.order} / {h.demand} / {h.sales} / {h.lost}")
    inbox = [n for n in obs.notes if n.text]
    if inbox:
        lines.append("Inbox:")
        for n in inbox[-6:]:
            when = "this week" if n.week == obs.week else f"week {n.week + 1}"
            lines.append(f"- ({when}) {n.text}")
    else:
        lines.append("Inbox: nothing new.")
    return "\n".join(lines)


def order_question(w: WorldConfig) -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": "How many units should the planner order this week to maximise profit over the rest of the plan?",
        "criteria": {order_key(q): (f"Order {q} units" if q else "Order nothing this week") for q in w.order_menu},
    }


def regime_question() -> dict[str, Any]:
    return {
        "type": "choice",
        "instructions": "What is customer demand most likely doing this week?",
        "criteria": {"low": "Slower than usual", "normal": "About usual", "surge": "Much higher than usual"},
    }


def request(w: WorldConfig, obs: Observation, context: str | None = None) -> dict[str, Any]:
    """The System One request for this week (no labels)."""
    return {
        "state": state_text(w, obs, context),
        "questions": {ORDER_Q: order_question(w), REGIME_Q: regime_question()},
    }


def change_question() -> dict[str, Any]:
    """The weekly alarm (H6): a yes/no question, scored as P(true)."""
    return {
        "type": "noul",
        "instructions": "Has something changed in the business that the usual planning playbook does not account for?",
    }


def question_keys(q: dict[str, Any]) -> list[str]:
    """Option keys in the order a policy scores them: the criteria for a choice, false/true for a yes/no (noul)."""
    return ["false", "true"] if q["type"] == "noul" else list(q["criteria"])
