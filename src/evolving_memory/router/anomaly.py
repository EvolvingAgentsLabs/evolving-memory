"""AnomalyDetector — detects semantic drift during graph traversal."""

from __future__ import annotations

import numpy as np

from ..embeddings.encoder import EmbeddingEncoder


class AnomalyDetector:
    """Detects when the current context has drifted too far from the traversal goal."""

    def __init__(self, encoder: EmbeddingEncoder, threshold: float = 0.3) -> None:
        self._encoder = encoder
        self._threshold = threshold

    def check(self, goal_text: str, current_context: str) -> tuple[bool, float]:
        """Returns (anomaly_detected, similarity_score).

        An anomaly is detected when the cosine similarity between the goal
        and the current context drops below the threshold.
        """
        goal_vec = self._encoder.encode(goal_text)
        context_vec = self._encoder.encode(current_context)

        # Cosine similarity
        goal_norm = goal_vec / (np.linalg.norm(goal_vec) + 1e-9)
        context_norm = context_vec / (np.linalg.norm(context_vec) + 1e-9)
        similarity = float(np.dot(goal_norm, context_norm))

        return similarity < self._threshold, similarity
