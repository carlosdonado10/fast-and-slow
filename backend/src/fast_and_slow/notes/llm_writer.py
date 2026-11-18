"""GPU note writer: an open instruct model (Phi-4-mini by default, MIT licence, a different family from Kev's Qwen base)
writes realistic planner notes from each cell's hidden attributes.

The writer only ever sees the attributes (source type, claim direction); it never sees regimes, outcomes or agents. The
text is generated once, deduplicated, filtered and split, then frozen in the bank.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import NotesConfig, WorldConfig
from ..progress import Progress
from .bank import NoteBank, cell_spec, split_cells

SOURCE_BRIEF = {
    "rumour": "an unverified rumour or piece of hearsay (trade chatter, a competitor's remark). Hedge it.",
    "sales_rep": "a field sales rep's update about one named retail account. Fairly confident, informal.",
    "customer_order": "a firm, written confirmation from a named customer (PO change, committed forecast). Factual.",
}
CLAIM_BRIEF = {
    "down": "demand for our product is about to be LOWER than usual",
    "flat": "demand will be about the SAME as usual",
    "up": "demand for our product is about to be HIGHER than usual",
}
SYSTEM = (
    "You write short internal notes that land in a demand planner's inbox at a consumer-goods supplier. "
    "Each note is one or two sentences, plain and realistic, with invented but plausible retailer names. "
    "Never mention probabilities, regimes or numbers of units."
)


def _prompt(spec: dict[str, Any], n: int) -> str:
    if spec["kind"] == "distractor":
        task = "Each note is office or operations noise that says NOTHING about customer demand (facilities, IT, HR, admin)."
    else:
        task = f"Each note is {SOURCE_BRIEF[spec['source']]} It implies that {CLAIM_BRIEF[spec['claim']]}."
    return (
        f"Write {n} different notes. {task} Vary the wording, retailer and reason. Return a JSON list of strings only."
    )


def _parse(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            items = json.loads(m.group(0))
            return [s for s in items if isinstance(s, str)]
        except json.JSONDecodeError:
            pass
    return [ln.strip(' -*0123456789."') for ln in text.splitlines() if len(ln.strip()) > 20]


def write(world: WorldConfig, cfg: NotesConfig, batch: int = 12, max_rounds: int = 12) -> NoteBank:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.llm_model)
    model = AutoModelForCausalLM.from_pretrained(cfg.llm_model, dtype=torch.bfloat16, device_map="cuda")
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(cfg.seed)

    raw: dict[str, list[str]] = {}
    specs = cell_spec(world)
    prog = Progress("notes (Phi-4-mini)", len(specs) * cfg.per_cell, "notes", every=0)
    for r in range(max_rounds):
        prog.update(
            sum(min(len(set(raw.get(c, []))), cfg.per_cell) for c in specs), f"round {r + 1} of up to {max_rounds}"
        )
        todo = [c for c in specs if len(set(raw.get(c, []))) < cfg.per_cell]
        if not todo:
            break
        chats = [
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _prompt(specs[c], batch)}] for c in todo
        ]
        enc = tok.apply_chat_template(
            chats, add_generation_prompt=True, return_tensors="pt", padding=True, return_dict=True
        ).to("cuda")
        out = model.generate(**enc, max_new_tokens=900, do_sample=True, temperature=0.9, top_p=0.95)
        for c, seq in zip(todo, out, strict=True):
            text = tok.decode(seq[enc["input_ids"].shape[1] :], skip_special_tokens=True)
            notes = [n for n in _parse(text) if 20 <= len(n) <= 280]
            raw.setdefault(c, []).extend(notes)
    raw = {c: list(dict.fromkeys(v))[: cfg.per_cell] for c, v in raw.items()}
    prog.update(sum(len(v) for v in raw.values()), force=True)
    short = {c: len(v) for c, v in raw.items() if len(v) < cfg.per_cell // 2}
    if short:
        raise RuntimeError(f"note writer produced too few notes for {short}; raise max_rounds")
    del model
    torch.cuda.empty_cache()
    return NoteBank(split_cells(raw, cfg), writer=cfg.llm_model)
