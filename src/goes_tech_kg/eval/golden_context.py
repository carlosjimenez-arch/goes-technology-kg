"""Context lookups built from the committed real-source excerpts (tests/golden)."""

import base64
import json
from pathlib import Path
from typing import Any

from goes_tech_kg.corpus.chunk import chunk_document
from goes_tech_kg.corpus.parse import parse
from goes_tech_kg.retrieval.context import ChunkLookup, chunk_from_model
from goes_tech_kg.schemas.corpus import Chunk, SourceDocument


def golden_chunks(golden_dir: Path) -> dict[str, tuple[SourceDocument, tuple[Chunk, ...]]]:
    """Parse the committed real-source excerpts into chunks, keyed by document slug."""
    result: dict[str, tuple[SourceDocument, tuple[Chunk, ...]]] = {}
    for path in sorted(golden_dir.glob("*.json")):
        row = json.loads(path.read_text())
        if "payload_base64" not in row:
            continue
        document = SourceDocument.model_validate(row["document"])
        payload = base64.b64decode(row["payload_base64"])
        result[document.slug] = (document, chunk_document(parse(document, payload)))
    return result


def golden_lookup(golden_dir: Path) -> tuple[ChunkLookup, dict[str, str]]:
    """Chunk lookup plus document-id to slug map, derived only from committed fixtures."""
    rows: dict[str, dict[str, Any]] = {}
    slugs: dict[str, str] = {}
    for slug, (document, chunks) in golden_chunks(golden_dir).items():
        slugs[document.id] = slug
        for chunk in chunks:
            rows[chunk.id] = chunk_from_model(chunk)

    def lookup(chunk_id: str) -> dict[str, Any]:
        if chunk_id not in rows:
            raise KeyError(f"chunk not in golden fixtures: {chunk_id}")
        return rows[chunk_id]

    return lookup, slugs
