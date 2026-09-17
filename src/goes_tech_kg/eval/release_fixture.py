"""Build an offline release test corpus exclusively from the attributed real golden excerpts."""

import base64
import json
from pathlib import Path

import yaml

from goes_tech_kg.corpus.chunk import chunk_document
from goes_tech_kg.corpus.license_gate import license_gate
from goes_tech_kg.corpus.parse import parse
from goes_tech_kg.retrieval.release_context import EvidenceIndex
from goes_tech_kg.schemas.base import byte_digest, canonical_json
from goes_tech_kg.schemas.corpus import Acquisition, Chunk, SourceDocument
from goes_tech_kg.schemas.release import UnitPlan


def release_fixture(root: Path, data: Path) -> tuple[EvidenceIndex, UnitPlan]:
    policy = yaml.safe_load((root / "corpus/allowlist.yaml").read_text())
    (data / "raw/corpus").mkdir(parents=True, exist_ok=True)
    (data / "interim").mkdir(exist_ok=True)
    (data / "manifests").mkdir(exist_ok=True)
    records = []
    chunks: list[Chunk] = []
    source_units = []
    for name in ("eng-dt", "eng-computing"):
        row = json.loads((root / "tests/golden" / f"{name}.json").read_text())
        document = SourceDocument.model_validate(row["document"])
        payload = base64.b64decode(row["payload_base64"])
        sha = byte_digest(payload)
        if sha != row["excerpt_sha256"]:
            raise ValueError("golden excerpt changed")
        (data / "raw/corpus" / sha).write_bytes(payload)
        record = Acquisition(
            document=document,
            license_decision=license_gate(document, policy),
            status="accepted",
            sha256=sha,
            byte_count=len(payload),
            acquired_at=row["acquired_at"],
            final_url=document.url,
        )
        records.append(record)
        unit_chunks = chunk_document(parse(document, payload))
        chunks.extend(unit_chunks)
        source_units.append((document.slug, unit_chunks[0].unit_id))
    (data / "manifests/corpus.jsonl").write_text("".join(canonical_json(r) + "\n" for r in records))
    (data / "interim/chunks.jsonl").write_text("".join(canonical_json(c) + "\n" for c in chunks))
    metadata = data / "processed/ingestion"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "chunks.jsonl").write_text(
        "".join(
            canonical_json({"id": c.id, "text_sha256": byte_digest(c.text.encode())}) + "\n"
            for c in chunks
        )
    )
    plan = UnitPlan(
        id="g2-release-golden-design",
        grade=2,
        strand="design_process",
        title="Design a purposeful artifact",
        learning_goal="Identify a need, sketch and make an artifact, and evaluate it against its intended purpose.",
        progression="Move from identifying a need to testing and improving a simple design.",
        retrieval_query="design make evaluate purpose materials tools",
        source_units=tuple(source_units),
        order=1,
    )
    return EvidenceIndex(data), plan
