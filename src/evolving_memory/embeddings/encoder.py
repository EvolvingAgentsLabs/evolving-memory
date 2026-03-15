"""Embedding encoder — Gemini Embedding 2 wrapper for semantic pointers."""

from __future__ import annotations

import os

import numpy as np

from google import genai
from google.genai import types


class EmbeddingEncoder:
    """Encodes text into dense vectors using Gemini Embedding 2."""

    def __init__(
        self,
        model_name: str = "gemini-embedding-2-preview",
        dim: int = 768,
        api_key: str | None = None,
    ) -> None:
        self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))
        self._model = model_name
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a dense vector."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=self._dim,
            ),
        )
        vec = np.array(result.embeddings[0].values, dtype=np.float32)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of text strings."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=self._dim,
            ),
        )
        vecs = np.array([e.values for e in result.embeddings], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        return vecs / norms
