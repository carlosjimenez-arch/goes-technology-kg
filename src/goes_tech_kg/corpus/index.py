"""Exact dense-vector index with deterministic row ordering and tie breaks."""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from goes_tech_kg.schemas.base import byte_digest, canonical_json


class VectorIndex:
    def __init__(self, ids: tuple[str, ...], vectors: NDArray[np.float32]):
        if len(set(ids)) != len(ids) or vectors.ndim != 2 or len(ids) != len(vectors):
            raise ValueError("duplicate IDs or inconsistent matrix shape")
        if not np.isfinite(vectors).all():
            raise ValueError("non-finite vectors")
        order = np.argsort(np.asarray(ids), kind="stable")
        self.ids = tuple(ids[int(i)] for i in order)
        self.vectors = np.asarray(vectors[order], dtype="<f4")

    def search(self, query: NDArray[np.float32], limit: int = 5) -> list[tuple[str, float]]:
        if query.shape != (self.vectors.shape[1],) or not np.isfinite(query).all() or limit < 1:
            raise ValueError("invalid query")
        scores = self.vectors @ query
        order = np.lexsort((np.asarray(self.ids), -scores))[:limit]
        return [(self.ids[int(i)], float(scores[i])) for i in order]

    def save(self, path: Path) -> dict[str, object]:
        path.mkdir(parents=True, exist_ok=True)
        matrix = self.vectors.tobytes(order="C")
        (path / "vectors.f32").write_bytes(matrix)
        metadata: dict[str, object] = {
            "schema_version": "vector-index/1.0",
            "ids": self.ids,
            "shape": self.vectors.shape,
            "dtype": "<f4",
            "sha256": byte_digest(matrix),
        }
        (path / "index.json").write_text(canonical_json(metadata) + "\n")
        return metadata

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        metadata = json.loads((path / "index.json").read_text())
        body = (path / "vectors.f32").read_bytes()
        if byte_digest(body) != metadata["sha256"]:
            raise ValueError("corrupt index")
        vectors = np.frombuffer(body, dtype="<f4").reshape(metadata["shape"])
        return cls(tuple(metadata["ids"]), vectors)
