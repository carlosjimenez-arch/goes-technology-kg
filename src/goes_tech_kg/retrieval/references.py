"""Build context references from the local chunk store.

A `ContextRef` names evidence by locator, never by text, so a plan can be committed while the
source wording stays in ignored interim storage. This module is the only place that knows how
to turn a document slug and a unit label or paragraph key into those locators.
"""

import json
from pathlib import Path
from typing import Any

from goes_tech_kg.retrieval.context import ChunkLookup
from goes_tech_kg.schemas.corpus import Acquisition
from goes_tech_kg.schemas.llm import ContextRef


class ChunkStore:
    """Index over the interim chunk file, addressable by chunk id or by document and unit."""

    def __init__(self, chunks_path: Path, manifest: tuple[Acquisition, ...] = ()):
        self.by_id: dict[str, dict[str, Any]] = {}
        with chunks_path.open() as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    self.by_id[str(row["id"])] = row
        self.by_unit = {(r["document_id"], r["unit_id"]): r for r in self.by_id.values()}
        self.documents = {record.document.slug: record.document for record in manifest}

    @property
    def lookup(self) -> ChunkLookup:
        """Chunk lookup for context assembly; an unknown id fails rather than returning empty."""

        def resolve(chunk_id: str) -> dict[str, Any]:
            if chunk_id not in self.by_id:
                raise KeyError(f"chunk not available locally: {chunk_id}")
            return self.by_id[chunk_id]

        return resolve

    @property
    def document_slugs(self) -> dict[str, str]:
        """Document id to slug, the map a plan needs to render locators."""
        return {document.id: slug for slug, document in self.documents.items()}

    def _paragraph_ids(self, row: dict[str, Any]) -> list[str]:
        return [str(p["original"]["id"]) for p in row["paragraphs"]]

    def whole_unit(self, slug: str, unit_label: str) -> ContextRef:
        """Reference every paragraph of one curricular unit, in document order."""
        document = self.documents[slug]
        row = self.by_unit[(document.id, unit_label)]
        return ContextRef(
            chunk_id=str(row["id"]),
            document_id=document.id,
            document_sha256=str(row["document_sha256"]),
            paragraph_ids=tuple(self._paragraph_ids(row)),
        )

    def paragraphs(self, slug: str, wanted: list[str]) -> tuple[ContextRef, ...]:
        """Reference selected paragraphs by short key, one reference per chunk they live in.

        Order follows `wanted`, so the assembled context reads in the order the case intends
        rather than in document order. Every requested key must resolve, or the plan is wrong.
        """
        document = self.documents[slug]
        order = {key: position for position, key in enumerate(wanted)}
        grouped: dict[str, list[str]] = {}
        for row in self.by_id.values():
            if row["document_id"] != document.id:
                continue
            for paragraph_id in self._paragraph_ids(row):
                if _short_key(paragraph_id) in order:
                    grouped.setdefault(str(row["id"]), []).append(paragraph_id)
        found = sum(len(v) for v in grouped.values())
        if found != len(wanted):
            raise ValueError(f"{slug}: expected {len(wanted)} paragraphs, resolved {found}")
        return tuple(
            ContextRef(
                chunk_id=chunk_id,
                document_id=document.id,
                document_sha256=str(self.by_id[chunk_id]["document_sha256"]),
                paragraph_ids=tuple(sorted(ids, key=lambda p: order[_short_key(p)])),
            )
            for chunk_id, ids in sorted(
                grouped.items(), key=lambda item: order[_short_key(item[1][0])]
            )
        )


def _short_key(paragraph_id: str) -> str:
    """The stable short key of a paragraph id, for example "p94" in "paragraph-p94-abc"."""
    return paragraph_id.split("-")[1]
