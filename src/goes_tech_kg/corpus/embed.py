"""Local multilingual embeddings with versioned, hash-checked offline replay."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from goes_tech_kg.schemas.base import byte_digest, canonical_json, digest

MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODEL_REVISION = "4328cf26390c98c5e3c738b4460a05b95f4911f5"


class Embedder:
    def __init__(self, model_path: Path | None, cache_root: Path, replay: bool = True):
        self.cache_root = cache_root
        self.replay = replay
        self.model: Any = None
        self.model_manifest: dict[str, str] = {}
        if model_path is not None:
            self.model_manifest = {
                str(p.relative_to(model_path)): byte_digest(p.read_bytes())
                for p in sorted(model_path.rglob("*"))
                if p.is_file() and p.suffix != ".md"
            }
            if not replay:
                import torch
                from sentence_transformers import SentenceTransformer

                torch.set_num_threads(1)
                torch.manual_seed(0)
                torch.use_deterministic_algorithms(True)
                self.model = SentenceTransformer(
                    str(model_path), device="cpu", local_files_only=True
                )
        manifest_path = cache_root / "model.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            if self.model_manifest and existing["files"] != self.model_manifest:
                raise ValueError("model snapshot changed")
            self.model_manifest = existing["files"]
        elif self.model_manifest and not replay:
            cache_root.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                canonical_json(
                    {"model_id": MODEL_ID, "revision": MODEL_REVISION, "files": self.model_manifest}
                )
                + "\n"
            )
        else:
            raise ValueError("missing versioned embedding model manifest")

    def embed(self, text: str) -> NDArray[np.float32]:
        key = digest(
            {
                "embedding_version": 1,
                "text_sha256": byte_digest(text.encode()),
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "files": self.model_manifest,
                "pooling": "token-window-mean-normalized",
                "threads": 1,
            }
        )
        path = self.cache_root / (key + ".json")
        if path.exists():
            record = json.loads(path.read_text())
            vector = np.asarray(record["vector"], dtype="<f4")
            if record["key"] != key or byte_digest(vector.tobytes()) != record["vector_sha256"]:
                raise ValueError("corrupt embedding replay")
            return vector
        if self.replay or self.model is None:
            raise ValueError(f"missing embedding replay: {key}")
        tokens = self.model.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, truncation=False
        )
        offsets = tokens["offset_mapping"]
        width = max(8, self.model.max_seq_length - 16)
        windows = [
            text[offsets[i][0] : offsets[min(i + width, len(offsets)) - 1][1]]
            for i in range(0, len(offsets), width)
        ] or [" "]
        if any(
            len(self.model.tokenizer(w)["input_ids"]) > self.model.max_seq_length for w in windows
        ):
            raise ValueError("embedding window would be truncated")
        vectors = np.asarray(
            self.model.encode(
                windows, batch_size=16, normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )
        vector = np.mean(vectors, axis=0, dtype=np.float64).astype("<f4")
        norm = float(np.linalg.norm(vector))
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("invalid embedding")
        vector = np.asarray(vector / norm, dtype="<f4")
        record = {
            "key": key,
            "text_sha256": byte_digest(text.encode()),
            "window_count": len(windows),
            "vector_sha256": byte_digest(vector.astype("<f4").tobytes()),
            "vector": vector.tolist(),
        }
        path.write_text(canonical_json(record) + "\n")
        return vector
