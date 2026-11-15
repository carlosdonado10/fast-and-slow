"""What the classic-RL agents see. Three variants, so the study can separate 'can read' from 'can learn':

- num: numbers only (week, stock, recent demand and orders), like planning software today
- emb: num + a text embedding of this week's note and a decayed memory of past notes (RL that can "read")
- sig: num + the note's parsed attributes (source x claim one-hot, plus a decayed memory): RL with a perfect parser
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..config import WorldConfig
from ..sim.desk import Observation
from ..sim.world import CLAIMS, SIGNAL

MEMORY_DECAY = 0.5
RECENT = 3


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic, dependency-free stand-in for a sentence encoder (smoke runs and tests)."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class CachedEmbedder:
    def __init__(self, inner: Embedder):
        self.inner, self.dim = inner, inner.dim
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            for t, v in zip(todo, self.inner.embed(todo), strict=True):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, self.dim))


def numeric(w: WorldConfig, obs: Observation) -> np.ndarray:
    scale = float(w.max_demand)
    recent = [h.demand / scale for h in obs.history[-RECENT:]]
    recent = [0.0] * (RECENT - len(recent)) + recent
    seen = [1.0] * min(len(obs.history), RECENT)
    seen = [0.0] * (RECENT - len(seen)) + seen
    last_order = obs.history[-1].order / max(w.order_menu) if obs.history else 0.0
    return np.asarray(
        [
            obs.week / obs.horizon,
            obs.weeks_left / obs.horizon,
            obs.inventory / w.inventory_cap,
            last_order,
            *recent,
            *seen,
        ],
        dtype=np.float32,
    )


def signal_onehot(w: WorldConfig, obs: Observation, week: int) -> np.ndarray:
    n = obs.notes[week].note
    v = np.zeros(len(w.sources) * len(CLAIMS) + 1, dtype=np.float32)
    if n.kind == SIGNAL:
        v[n.source * len(CLAIMS) + n.claim] = 1.0
    elif n.kind != "none":
        v[-1] = 1.0  # distractor
    return v


def features(w: WorldConfig, obs: Observation, variant: str, embedder: Embedder | None = None) -> np.ndarray:
    base = numeric(w, obs)
    if variant == "num":
        return base
    if variant == "sig":
        now = signal_onehot(w, obs, obs.week)
        mem = np.zeros_like(now)
        for t in range(obs.week):
            mem = mem * MEMORY_DECAY + signal_onehot(w, obs, t)
        return np.concatenate([base, now, mem]).astype(np.float32)
    if variant == "emb":
        if embedder is None:
            raise ValueError("emb features need an embedder")
        texts = [n.text for n in obs.notes]
        vecs = embedder.embed([t or "" for t in texts])
        vecs[[i for i, t in enumerate(texts) if not t]] = 0.0
        mem = np.zeros(embedder.dim)
        for t in range(obs.week):
            mem = mem * MEMORY_DECAY + vecs[t]
        return np.concatenate([base, vecs[obs.week], mem]).astype(np.float32)
    raise ValueError(f"unknown feature variant {variant!r}")


def feature_dim(w: WorldConfig, variant: str, embed_dim: int = 0) -> int:
    n = 4 + 2 * RECENT
    sig = len(w.sources) * len(CLAIMS) + 1
    return {"num": n, "sig": n + 2 * sig, "emb": n + 2 * embed_dim}[variant]
