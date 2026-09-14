"""Resolve graph evidence all the way back to a hash-verified source paragraph."""

from pathlib import Path

from goes_tech_kg.corpus.fetch import replay_bytes
from goes_tech_kg.corpus.normalize import normalize, original_range
from goes_tech_kg.corpus.parse import parse
from goes_tech_kg.schemas.corpus import Acquisition
from goes_tech_kg.schemas.skills import EvidenceRef


def resolve_evidence(ref: EvidenceRef, record: Acquisition, raw_root: Path) -> dict[str, object]:
    if ref.document_id != record.document.id or ref.document_sha256 != record.sha256:
        raise ValueError("evidence document/digest mismatch")
    parsed = parse(record.document, replay_bytes(record, raw_root))
    paragraphs = {p.id: p for p in parsed.paragraphs}
    if ref.paragraph_id not in paragraphs:
        raise ValueError("unknown source paragraph")
    paragraph = normalize(paragraphs[ref.paragraph_id])
    start, end = original_range(paragraph, ref.start, ref.end)
    return {
        "document_url": str(record.document.url),
        "document_sha256": record.sha256,
        "page": paragraph.original.page,
        "section": paragraph.original.section,
        "dom_path": paragraph.original.dom_path,
        "paragraph_id": ref.paragraph_id,
        "normalized_text": paragraph.text[ref.start : ref.end],
        "original_text": paragraph.original.text[start:end],
        "original_start": start,
        "original_end": end,
    }
