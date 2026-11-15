"""The note bank: paraphrases for every (source, claim) cell plus distractors, split into train and test wording.

The bank is written once (templates offline, or an open LLM on the GPU) and saved as JSON, so every arm and every run
reads the same notes. Evaluation always uses the *test* paraphrases, which no agent trained on.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import NotesConfig, WorldConfig
from ..sim.world import CLAIMS, DISTRACTOR, NONE, Note, Scenario


@dataclass
class NoteBank:
    cells: dict[str, dict[str, list[str]]]  # cell -> {"train": [...], "test": [...]}
    writer: str

    def text(self, note: Note, split: str) -> str | None:
        if note.kind == NONE:
            return None
        pool = self.cells[note.cell][split]
        return pool[note.variant % len(pool)]

    def texts(self, scenario: Scenario, split: str) -> list[str | None]:
        return [self.text(n, split) for n in scenario.notes]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"writer": self.writer, "cells": self.cells}, indent=1, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> NoteBank:
        data = json.loads(path.read_text())
        return cls(data["cells"], data["writer"])


def cell_spec(world: WorldConfig) -> dict[str, dict[str, Any]]:
    """The hidden attributes each cell's notes must express (what the writer is told)."""
    spec: dict[str, dict[str, Any]] = {DISTRACTOR: {"kind": DISTRACTOR}}
    for s, src in enumerate(world.sources):
        for c, claim in enumerate(CLAIMS):
            spec[f"{s}:{c}"] = {"kind": "signal", "source": src, "claim": claim, "accuracy": world.source_accuracy[s]}
    return spec


def split_cells(raw: dict[str, list[str]], cfg: NotesConfig) -> dict[str, dict[str, list[str]]]:
    rng = random.Random(cfg.seed)
    out = {}
    for cell, texts in raw.items():
        texts = list(dict.fromkeys(t.strip() for t in texts if t.strip()))  # dedupe, keep order
        rng.shuffle(texts)
        n_test = max(1, round(len(texts) * cfg.test_fraction))
        out[cell] = {"train": texts[n_test:] or texts, "test": texts[:n_test]}
    return out
