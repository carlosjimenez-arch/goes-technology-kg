"""Validate the actual recorded Vertex embedding, including corrupted/truncated replay."""

import json
from pathlib import Path

import numpy as np
import pytest

from goes_tech_kg.llm.embeddings import VertexEmbeddings

ROOT = Path(__file__).resolve().parents[2]


def test_real_cloud_embedding_replays_without_credentials(tmp_path):
    query = json.loads((ROOT / "evaluation/retrieval-v2/plan.json").read_text())["queries"][0][
        "query"
    ]
    cache = ROOT / "data/processed/retrieval_vertex_embeddings"
    engine = VertexEmbeddings(cache, "unused-offline")
    vector = engine.embed(query, "RETRIEVAL_QUERY")
    assert vector.shape == (768,) and np.isclose(np.linalg.norm(vector), 1)
    assert engine.acquired == 0
    with pytest.raises(ValueError, match="missing cloud embedding"):
        VertexEmbeddings(tmp_path, "unused-offline").embed(query, "RETRIEVAL_QUERY")
    for path in cache.glob("*.json"):
        row = json.loads(path.read_text())
        row["response_sha256"] = "0" * 64
        (tmp_path / path.name).write_text(json.dumps(row))
    with pytest.raises(ValueError, match="corrupt cloud embedding"):
        VertexEmbeddings(tmp_path, "unused-offline").embed(query, "RETRIEVAL_QUERY")


def test_recorded_vector_rejects_truncation_and_invalid_dimensions():
    path = next((ROOT / "data/processed/retrieval_vertex_embeddings").glob("*.json"))
    payload = json.loads(path.read_text())["response"]
    embedding = payload["predictions"][0]["embeddings"]
    embedding["statistics"]["truncated"] = True
    with pytest.raises(ValueError, match="truncated"):
        VertexEmbeddings._vector(payload)
    embedding["statistics"]["truncated"] = False
    embedding["values"] = embedding["values"][:10]
    with pytest.raises(ValueError, match="invalid cloud embedding vector"):
        VertexEmbeddings._vector(payload)
