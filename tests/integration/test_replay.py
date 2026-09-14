import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from goes_tech_kg.corpus.embed import Embedder
from goes_tech_kg.corpus.fetch import restore
from goes_tech_kg.corpus.license_gate import license_gate
from goes_tech_kg.corpus.parse import parse
from goes_tech_kg.corpus.trace import resolve_evidence
from goes_tech_kg.schemas.base import byte_digest
from goes_tech_kg.schemas.corpus import Acquisition
from goes_tech_kg.schemas.skills import EvidenceRef


def test_real_source_evidence_resolves_and_rejects_tampering(golden_documents, tmp_path):
    policy = yaml.safe_load(Path("corpus/allowlist.yaml").read_text())
    for document, payload, _ in golden_documents:
        sha = byte_digest(payload)
        (tmp_path / sha).write_bytes(payload)
        record = Acquisition(
            document=document,
            license_decision=license_gate(document, policy),
            status="accepted",
            sha256=sha,
            byte_count=len(payload),
            acquired_at=datetime(2026, 9, 13, tzinfo=UTC),
            final_url=document.url,
        )
        paragraph = next(p for p in parse(document, payload).paragraphs if len(p.text.strip()) > 10)
        ref = EvidenceRef(
            document_id=document.id, document_sha256=sha, paragraph_id=paragraph.id, start=0, end=10
        )
        resolved = resolve_evidence(ref, record, tmp_path)
        assert resolved["original_text"] == paragraph.text[:10]
        assert resolved["page"] or resolved["dom_path"]
        with httpx.Client() as client:
            restore(record, policy, tmp_path, client)  # Offline, already pinned; sockets disabled.
        (tmp_path / sha).write_bytes(payload + b"corruption")
        with pytest.raises(ValueError, match="digest"):
            resolve_evidence(ref, record, tmp_path)


def test_real_embedding_cache_replays_and_detects_modified_vectors(tmp_path):
    cache = Path("data/processed/embeddings")
    (tmp_path / "model.json").write_bytes((cache / "model.json").read_bytes())
    embedder = Embedder(None, tmp_path)
    with pytest.raises(ValueError, match="missing embedding replay"):
        embedder.embed("An intentionally uncached input")
    # Compare an actual recorded embedding to the corresponding reconstructed source chunk.
    from goes_tech_kg.corpus.chunk import chunk_document

    document, payload, _ = next(iter_real_html())
    chunk = chunk_document(parse(document, payload))[0]
    original = Embedder(None, cache).embed(chunk.text)
    record_path = next(
        p
        for p in cache.glob("*.json")
        if p.name != "model.json"
        and json.loads(p.read_text())["text_sha256"] == byte_digest(chunk.text.encode())
    )
    target = tmp_path / record_path.name
    target.write_bytes(record_path.read_bytes())
    np.testing.assert_array_equal(embedder.embed(chunk.text), original)
    row = json.loads(target.read_text())
    row["vector"][0] += 1
    target.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="corrupt embedding"):
        embedder.embed(chunk.text)


def iter_real_html():
    # The whole activity fixture retains the selected DOM subtree; it produces the same embedding text.
    import base64

    from goes_tech_kg.schemas.corpus import SourceDocument

    row = json.loads(Path("tests/golden/unplugged-rocket.json").read_text())
    yield (
        SourceDocument.model_validate(row["document"]),
        base64.b64decode(row["payload_base64"]),
        row,
    )


def test_real_excerpt_pipeline_replays_byte_identically_and_reports_missing_coverage(
    golden_documents, tmp_path
):
    from goes_tech_kg.corpus.pipeline import compile_corpus
    from goes_tech_kg.schemas.base import canonical_json

    raw = tmp_path / "raw/corpus"
    raw.mkdir(parents=True)
    manifest = tmp_path / "manifests/corpus.jsonl"
    manifest.parent.mkdir()
    policy = yaml.safe_load(Path("corpus/allowlist.yaml").read_text())
    rows = []
    for document, payload, _ in golden_documents:
        sha = byte_digest(payload)
        (raw / sha).write_bytes(payload)
        rows.append(
            Acquisition(
                document=document,
                license_decision=license_gate(document, policy),
                status="accepted",
                sha256=sha,
                byte_count=len(payload),
                acquired_at=datetime(2026, 9, 13, tzinfo=UTC),
                final_url=document.url,
            )
        )
    manifest.write_text("".join(canonical_json(row) + "\n" for row in rows))
    first = compile_corpus(tmp_path)
    saved = {
        str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
    second = compile_corpus(tmp_path)
    assert first == second and first["chunks"] == 3
    assert not first["coverage_complete"] and not first["index_built"]
    assert all(status["status"] == "parsed" for status in first["statuses"])
    assert saved == {
        str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }
