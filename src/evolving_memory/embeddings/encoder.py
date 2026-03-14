"""Embedding encoder — sentence-transformers wrapper for semantic pointers."""

from __future__ import annotations

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment, misc]


class EmbeddingEncoder:
    """Encodes text into dense vectors using a sentence-transformers model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required: pip install sentence-transformers")
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a dense vector."""
        return self._model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of text strings."""
        return self._model.encode(texts, convert_to_numpy=True, batch_size=32)
