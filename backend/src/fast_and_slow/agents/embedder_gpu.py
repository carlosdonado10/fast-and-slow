"""Sentence-transformer embedder for the `emb` PPO variant (GPU extra; excluded from coverage)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class SentenceEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, batch_size=128))
