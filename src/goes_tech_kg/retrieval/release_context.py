"""Deterministic, source-diverse paragraph retrieval and exact citation resolution."""

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

from goes_tech_kg.corpus.normalize import original_range
from goes_tech_kg.corpus.pipeline import read_manifest
from goes_tech_kg.retrieval.context import compact, locator
from goes_tech_kg.schemas.base import byte_digest
from goes_tech_kg.schemas.corpus import Chunk, NormalizedParagraph
from goes_tech_kg.schemas.llm import ContextRef
from goes_tech_kg.schemas.release import UnitPlan
from goes_tech_kg.schemas.skills import EvidenceRef

STOP = frozenset(
    "the and with for that this from into their have they are como para por una los las del que con sus estos esta".split()
)


def tokens(text: str) -> set[str]:
    folded = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
    return set(re.findall(r"[a-z]{3,}", folded)) - STOP


class EvidenceIndex:
    def __init__(self, data_root: Path):
        self.records = read_manifest(data_root / "manifests/corpus.jsonl")
        self.slugs = {r.document.id: r.document.slug for r in self.records}
        accepted = {r.document.id: r.sha256 for r in self.records if r.status == "accepted"}
        metadata_path = data_root / "processed/ingestion/chunks.jsonl"
        metadata = {
            r["id"]: r
            for r in (json.loads(line) for line in metadata_path.read_text().splitlines())
        }
        self.chunks: dict[str, dict[str, Any]] = {}
        self.paragraphs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for line in (data_root / "interim/chunks.jsonl").read_text().splitlines():
            chunk = json.loads(line)
            Chunk.model_validate(chunk)
            expected = metadata.get(chunk["id"])
            if expected is None or expected["text_sha256"] != byte_digest(chunk["text"].encode()):
                raise ValueError("interim text differs from versioned ingestion metadata")
            if accepted.get(chunk["document_id"]) != chunk["document_sha256"]:
                raise ValueError("interim chunk lacks accepted provenance")
            if (
                byte_digest((data_root / "raw/corpus" / chunk["document_sha256"]).read_bytes())
                != chunk["document_sha256"]
            ):
                raise ValueError("raw source hash mismatch")
            self.chunks[chunk["id"]] = chunk
            for paragraph in chunk["paragraphs"]:
                self.paragraphs[paragraph["original"]["id"]] = (chunk, paragraph)

    def select(self, unit: UnitPlan, per_source: int = 4) -> tuple[ContextRef, ...]:
        candidates = []
        wanted = set(unit.source_units)
        present = {(self.slugs[c["document_id"]], c["unit_id"]) for c in self.chunks.values()}
        if not wanted.issubset(present):
            raise ValueError(f"unknown source units: {sorted(wanted - present)}")
        query = tokens(unit.retrieval_query + " " + unit.learning_goal)
        for cid, chunk in sorted(self.chunks.items()):
            source = (self.slugs[chunk["document_id"]], chunk["unit_id"])
            if source not in wanted:
                continue
            scored = []
            for p in chunk["paragraphs"]:
                if self.slugs[chunk["document_id"]] == "eng-computing" and p["original"][
                    "section"
                ] in {"Key stage 3", "Key stage 4"}:
                    continue
                text = compact(p["text"])
                if not 40 <= len(text) <= 9000:
                    continue
                terms = tokens(text)
                score = len(query & terms) / math.sqrt(max(1, len(terms)))
                scored.append((-score, p["original"]["id"]))
            ids = tuple(pid for _, pid in sorted(scored)[:per_source])
            if ids:
                candidates.append(
                    ContextRef(
                        chunk_id=cid,
                        document_id=chunk["document_id"],
                        document_sha256=chunk["document_sha256"],
                        paragraph_ids=ids,
                    )
                )
        if len(candidates) < 2:
            raise ValueError("release context requires two source units")
        return tuple(candidates)

    def render(self, refs: tuple[ContextRef, ...]) -> str:
        blocks = []
        for ref in refs:
            for pid in ref.paragraph_ids:
                chunk, paragraph = self.paragraphs[pid]
                if chunk["id"] != ref.chunk_id or chunk["document_sha256"] != ref.document_sha256:
                    raise ValueError("context reference changed")
                blocks.append(
                    locator(self.slugs[ref.document_id], paragraph)
                    + " "
                    + compact(paragraph["text"])
                )
        return "\n".join(blocks)

    def resolve_quote(self, quote: str, hint: str, refs: tuple[ContextRef, ...]) -> EvidenceRef:
        # Search only the cited paragraph. A quote found elsewhere is not a correct citation.
        allowed = {pid for ref in refs for pid in ref.paragraph_ids}
        candidates = [pid for pid in sorted(allowed) if pid in hint]
        if len(candidates) != 1:
            raise ValueError("citation must identify exactly one supplied paragraph")
        pid = candidates[0]
        chunk, paragraph = self.paragraphs[pid]
        if self.slugs[chunk["document_id"]] == "eng-computing" and paragraph["original"][
            "section"
        ] in {"Key stage 3", "Key stage 4"}:
            raise ValueError("secondary-stage evidence cannot justify this primary curriculum")
        pattern = r"\s+".join(re.escape(part) for part in quote.split())
        match = re.search(pattern, paragraph["text"])
        if match is None:
            raise ValueError(
                f"unsupported quotation in {pid}; use a shorter exact fragment from one contiguous source line, without dehyphenation or interleaved table labels"
            )
        start, end = original_range(
            NormalizedParagraph.model_validate(paragraph), match.start(), match.end()
        )
        return EvidenceRef(
            document_id=chunk["document_id"],
            document_sha256=chunk["document_sha256"],
            paragraph_id=pid,
            start=start,
            end=end,
        )
