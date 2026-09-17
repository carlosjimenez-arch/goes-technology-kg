"""Deterministic context assembly from chunk locators; every paragraph carries its locator."""

import json
import re
from collections.abc import Callable
from typing import Any

from goes_tech_kg.schemas.llm import ContextRef

ChunkLookup = Callable[[str], dict[str, Any]]
_SPACES = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def compact(text: str) -> str:
    """Collapse layout whitespace while preserving line structure and every word."""
    return _BLANK_LINES.sub("\n", _SPACES.sub(" ", text)).strip()


def chunk_from_model(chunk: Any) -> dict[str, Any]:
    """Adapt a Chunk contract (for example from a golden excerpt) to the lookup row shape."""
    row: dict[str, Any] = json.loads(chunk.model_dump_json())
    return row


def locator(document_slug: str, paragraph: dict[str, Any]) -> str:
    """The bracketed reference a model must cite: document, page or html, paragraph id."""
    original = paragraph["original"]
    where = f"p.{original['page']}" if original.get("page") is not None else "html"
    return f"[{document_slug} {where} ¶{original['id']}]"


def assemble_context(
    refs: tuple[ContextRef, ...], lookup: ChunkLookup, slugs: dict[str, str]
) -> str:
    """Join the referenced paragraphs in reference order, each prefixed with its locator."""
    blocks: list[str] = []
    for ref in refs:
        chunk = lookup(ref.chunk_id)
        if chunk["document_sha256"] != ref.document_sha256:
            raise ValueError(f"chunk digest changed: {ref.chunk_id}")
        by_id = {p["original"]["id"]: p for p in chunk["paragraphs"]}
        for pid in ref.paragraph_ids:
            if pid not in by_id:
                raise ValueError(f"paragraph {pid} not in chunk {ref.chunk_id}")
            paragraph = by_id[pid]
            text = compact(paragraph["text"])
            if text:
                blocks.append(f"{locator(slugs[ref.document_id], paragraph)} {text}")
    return "\n".join(blocks)
