"""FAISS vector index — semantic pointer/index for parent node entry point discovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]


class VectorIndex:
    """Wraps a FAISS flat-IP index for parent node embeddings."""

    def __init__(self, dim: int = 384, index_path: str | Path | None = None) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._id_map: list[str] = []  # position → node_id
        self._index_path = Path(index_path) if index_path else None
        if self._index_path and self._index_path.exists():
            self._load()

    # ── public API ──────────────────────────────────────────────────

    def add(self, node_id: str, vector: list[float] | np.ndarray) -> None:
        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        self._index.add(vec)
        self._id_map.append(node_id)

    def search(self, query_vector: list[float] | np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """Returns list of (node_id, similarity_score) sorted by score desc."""
        if self._index.ntotal == 0:
            return []
        vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    def remove(self, node_id: str) -> None:
        """Remove a node by rebuilding the index without it."""
        if node_id not in self._id_map:
            return
        idx = self._id_map.index(node_id)
        all_vecs = faiss.rev_swig_ptr(self._index.get_xb(), self._index.ntotal * self._dim)
        all_vecs = np.array(all_vecs).reshape(-1, self._dim)
        keep_mask = [i for i in range(len(self._id_map)) if i != idx]
        self._index.reset()
        self._id_map = [self._id_map[i] for i in keep_mask]
        if keep_mask:
            kept = all_vecs[keep_mask]
            self._index.add(kept)

    @property
    def size(self) -> int:
        return self._index.ntotal

    # ── persistence ─────────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> None:
        p = Path(path) if path else self._index_path
        if p is None:
            raise ValueError("No path specified for saving index")
        faiss.write_index(self._index, str(p))
        id_path = p.with_suffix(".ids")
        id_path.write_text("\n".join(self._id_map))

    def _load(self) -> None:
        if self._index_path is None:
            return
        self._index = faiss.read_index(str(self._index_path))
        id_path = self._index_path.with_suffix(".ids")
        if id_path.exists():
            self._id_map = id_path.read_text().strip().split("\n")
            if self._id_map == [""]:
                self._id_map = []
