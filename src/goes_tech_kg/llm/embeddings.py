"""Vertex embedding boundary with explicit truncation refusal and immutable vector replay."""

import json
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np
from numpy.typing import NDArray

from goes_tech_kg.llm.vertex import TokenProvider
from goes_tech_kg.schemas.base import byte_digest, canonical_json, digest


class VertexEmbeddings:
    def __init__(self, cache: Path, project: str, token_provider: TokenProvider | None = None):
        self.cache = cache
        self.project = project
        self.token_provider = token_provider
        self.acquired = 0

    def embed(
        self, text: str, task: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    ) -> NDArray[np.float32]:
        request = {
            "model": "gemini-embedding-001",
            "location": "us-central1",
            "task_type": task,
            "text_sha256": byte_digest(text.encode()),
            "dimensions": 768,
            "auto_truncate": False,
        }
        key = digest(request)
        path = self.cache / (key + ".json")
        if path.exists():
            row = json.loads(path.read_text())
            if row["request"] != request or digest(row["response"]) != row["response_sha256"]:
                raise ValueError("corrupt cloud embedding replay")
            payload = row["response"]
        else:
            if self.token_provider is None:
                raise ValueError("missing cloud embedding replay")
            if self.acquired >= 100:
                raise ValueError("embedding acquisition cap exceeded")
            url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{self.project}/locations/us-central1/publishers/google/models/gemini-embedding-001:predict"
            with httpx.Client(timeout=90) as client:
                response = client.post(
                    url,
                    headers={"Authorization": "Bearer " + self.token_provider()},
                    json={
                        "instances": [{"content": text, "task_type": task}],
                        "parameters": {"autoTruncate": False, "outputDimensionality": 768},
                    },
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Vertex embedding failed with status {response.status_code}"
                    )
                payload = response.json()
            self.acquired += 1
            vector = self._vector(payload)
            self.cache.mkdir(parents=True, exist_ok=True)
            path.write_text(
                canonical_json(
                    {"request": request, "response": payload, "response_sha256": digest(payload)}
                )
                + "\n"
            )
            return vector
        return self._vector(payload)

    @staticmethod
    def _vector(payload: dict[str, Any]) -> NDArray[np.float32]:
        embedding = payload["predictions"][0]["embeddings"]
        if embedding["statistics"].get("truncated"):
            raise ValueError("truncated cloud embedding rejected")
        values = np.asarray(embedding["values"], dtype="<f4")
        norm = float(np.linalg.norm(values))
        if values.shape != (768,) or not np.isfinite(norm) or norm == 0:
            raise ValueError("invalid cloud embedding vector")
        return np.asarray(values / norm, dtype="<f4")
