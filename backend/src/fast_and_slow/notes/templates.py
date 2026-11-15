"""Offline note writer: combinatorial templates. Used by the smoke profile and as the fallback when no GPU is around.

The full study uses notes/llm_writer.py (an open model writes the text). Both produce the same cells, so the rest of the
pipeline can't tell them apart.
"""

from __future__ import annotations

import itertools
import random

from ..config import NotesConfig, WorldConfig
from .bank import NoteBank, cell_spec, split_cells

ACCOUNTS = [
    "Hartwell Foods",
    "Marlow Grocers",
    "Kestrel Outdoor",
    "Bayside Pharmacy",
    "Ostrander Hardware",
    "Pinecrest Markets",
    "Tamsin & Co",
    "Northgate Wholesale",
    "Larkspur Retail",
    "Quarry Lane Stores",
]

OPENERS = {
    "rumour": [
        "Overheard at the distributor meeting:",
        "Unconfirmed, but",
        "Word on the trade floor is that",
        "A contact at a competitor mentioned that",
        "Hearing chatter that",
        "Grain of salt, but apparently",
    ],
    "sales_rep": [
        "From the account rep for {acct}:",
        "Field sales update ({acct}):",
        "Rep check-in with {acct}:",
        "Our rep covering {acct} reports that",
        "Sales note re {acct}:",
        "Quick one from the {acct} account team:",
    ],
    "customer_order": [
        "{acct} confirmed in writing that",
        "PO revision received from {acct}:",
        "Signed forecast from {acct}:",
        "{acct} procurement confirmed that",
        "EDI update from {acct}:",
        "{acct} sent their committed volumes:",
    ],
}

CLAIMS = {
    "down": [
        "volumes are softening for the next few weeks",
        "they are cutting back on orders",
        "sell-through has slowed noticeably",
        "they'll take less than usual this month",
        "demand looks weaker than plan",
        "they are working down their own stock first",
    ],
    "flat": [
        "volumes should stay in line with the usual run rate",
        "nothing unusual expected, business as usual",
        "they'll order about the same as last month",
        "demand is tracking plan",
        "no change to their typical pull",
        "steady weeks ahead, no promos planned",
    ],
    "up": [
        "a promotion is going live and volumes will jump",
        "they expect a big lift in demand",
        "they are expanding shelf space for us",
        "sell-through is running well ahead of plan",
        "they want more than usual over the coming weeks",
        "a competitor stock-out is sending customers their way",
    ],
}

DISTRACTORS = [
    "Reminder: the warehouse team moves to the new building on Friday.",
    "IT will patch the ERP system Sunday night; expect a short outage.",
    "Packaging redesign samples are in, feedback by Thursday please.",
    "Quarterly safety audit scheduled for the second week of next month.",
    "The forklift lease renewal is with procurement for signature.",
    "Marketing is refreshing the product photos on the website.",
    "Finance asks everyone to submit expenses before month end.",
    "New label printer installed on line 2, old one going to storage.",
    "Team lunch moved to Wednesday this week.",
    "Carrier invoices for last quarter have been reconciled.",
]


def write(world: WorldConfig, cfg: NotesConfig) -> NoteBank:
    rng = random.Random(cfg.seed)
    raw: dict[str, list[str]] = {}
    for cell, spec in cell_spec(world).items():
        if spec["kind"] == "distractor":
            pool = [f"{d}" for d in DISTRACTORS] + [f"FYI: {d[0].lower()}{d[1:]}" for d in DISTRACTORS]
        else:
            combos = list(itertools.product(OPENERS[spec["source"]], CLAIMS[spec["claim"]], ACCOUNTS))
            rng.shuffle(combos)
            pool = []
            for opener, claim, acct in combos[: cfg.per_cell * 2]:
                pool.append(f"{opener.format(acct=acct)} {claim}.")
        rng.shuffle(pool)
        raw[cell] = pool[: cfg.per_cell]
    return NoteBank(split_cells(raw, cfg), writer="templates")
