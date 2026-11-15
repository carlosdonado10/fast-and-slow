"""The decision-model policy interface the Kev arms and the RL trainer are written against.

A policy scores System One requests: for each request, one logit vector per question, over that question's options in
the order of its `criteria`. Probabilities are softmax(logits); for Kev the logits already include its fitted temperature.

Two implementations:
- ``kev_backend.KevPolicy``: the real Kev-4B (GPU, LoRA trainable). Not importable without the `gpu` extra.
- ``FakePolicy`` below: a small bag-of-hashed-words scorer with the same interface, so the trainer, the agents and the
  tests run anywhere. It knows nothing about planning; it only proves the plumbing.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

Request = dict[str, Any]


class DecisionPolicy(Protocol):
    name: str

    def score(self, requests: list[Request], grad: bool = False) -> list[dict[str, torch.Tensor]]:
        """Per request: {question id: logits over its options}. With grad=False runs under no_grad."""
        ...

    def param_groups(self) -> list[dict[str, Any]]:
        """Trainable parameter groups for the optimizer (lr set by the trainer from each group's 'role')."""
        ...

    def save(self, path: Path) -> None: ...


_TOKEN = re.compile(r"[a-z0-9$]+")


def _hash(token: str, buckets: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "little") % buckets


def hashed_bag(text: str, buckets: int) -> torch.Tensor:
    v = torch.zeros(buckets)
    toks = _TOKEN.findall(text.lower())
    following = [*toks[1:], ""] if toks else []
    for a, b in zip(toks, following, strict=True):
        v[_hash(a, buckets)] += 1.0
        if b:
            v[_hash(a + "_" + b, buckets)] += 1.0
    return v / max(1.0, float(v.norm()))


class FakePolicy(nn.Module):
    """score_k = state_features^T M option_features_k, with M low rank. Deterministic given the seed."""

    name = "fake-kev"

    def __init__(self, buckets: int = 512, rank: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.buckets = buckets
        self.state_proj = nn.Parameter(torch.randn(buckets, rank, generator=g) * 0.1)
        self.option_proj = nn.Parameter(torch.randn(buckets, rank, generator=g) * 0.1)
        self.head_bias = nn.Parameter(torch.zeros(1))

    def _logits(self, req: Request) -> dict[str, torch.Tensor]:
        s = hashed_bag(req["state"], self.buckets) @ self.state_proj
        out = {}
        for qid, q in req["questions"].items():
            crit = q["criteria"]
            texts = [f"{k} {v or ''} {q['instructions']}" for k, v in crit.items()]
            o = torch.stack([hashed_bag(t, self.buckets) for t in texts]) @ self.option_proj
            out[qid] = o @ s * 4.0 + self.head_bias
        return out

    def score(self, requests: list[Request], grad: bool = False) -> list[dict[str, torch.Tensor]]:
        with torch.set_grad_enabled(grad):
            return [self._logits(r) for r in requests]

    def param_groups(self) -> list[dict[str, Any]]:
        return [
            {"params": [self.state_proj, self.option_proj], "role": "adapter"},
            {"params": [self.head_bias], "role": "head"},
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "fake_policy.pt")


def load_policy(run: str, fake: bool, dtype: str = "bf16", seed: int = 0) -> DecisionPolicy:
    if fake:
        return FakePolicy(seed=seed)
    from .kev_backend import KevPolicy  # pragma: no cover - needs the gpu extra

    return KevPolicy(run, dtype=dtype)  # pragma: no cover
