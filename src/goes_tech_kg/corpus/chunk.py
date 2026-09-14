"""One chunk per contiguous curricular unit; no token truncation or overlap."""

from itertools import groupby

from goes_tech_kg.corpus.normalize import normalize
from goes_tech_kg.schemas.base import stable_id
from goes_tech_kg.schemas.corpus import Chunk, ParsedDocument


def chunk_document(parsed: ParsedDocument) -> tuple[Chunk, ...]:
    chunks = []
    for ordinal, (unit, group) in enumerate(groupby(parsed.paragraphs, key=lambda p: p.unit_id), 1):
        paragraphs = tuple(normalize(p) for p in group)
        chunks.append(
            Chunk(
                id=stable_id(
                    "chunk",
                    f"u{ordinal}",
                    {
                        "document": parsed.document_sha256,
                        "unit": unit,
                        "paragraphs": [p.original.id for p in paragraphs],
                        "normalizer": "crlf/1.0",
                    },
                ),
                document_id=parsed.document_id,
                document_sha256=parsed.document_sha256,
                unit_id=unit,
                paragraphs=paragraphs,
                text="".join(p.text for p in paragraphs),
            )
        )
    return tuple(chunks)
