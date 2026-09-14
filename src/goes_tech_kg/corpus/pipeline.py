"""Discover → gate → fetch → manifest → parse → normalize/chunk → embed → index."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import polars as pl
import yaml

from goes_tech_kg.agents.license_workflow import license_gate_node
from goes_tech_kg.corpus.chunk import chunk_document
from goes_tech_kg.corpus.discover import discover
from goes_tech_kg.corpus.embed import Embedder
from goes_tech_kg.corpus.fetch import fetch, replay_bytes
from goes_tech_kg.corpus.index import VectorIndex
from goes_tech_kg.corpus.parse import parse
from goes_tech_kg.corpus.provenance import build_provenance
from goes_tech_kg.schemas.base import byte_digest, canonical_json
from goes_tech_kg.schemas.corpus import Acquisition, Chunk, SourceDocument


def acquire(catalogue: Path, policy_path: Path, data_root: Path) -> tuple[Acquisition, ...]:
    sources = discover(catalogue)
    policy = yaml.safe_load(policy_path.read_text())
    with httpx.Client(
        timeout=45, headers={"User-Agent": "goes-tech-kg/0.1 curriculum research"}, verify=True
    ) as client:

        def get(source: SourceDocument) -> Acquisition:
            return fetch(
                source, license_gate_node(source, policy), policy, data_root / "raw/corpus", client
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            records = tuple(executor.map(get, sources))
    path = data_root / "manifests/corpus.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # History is immutable; a new acquisition never silently replaces the pinned manifest.
    body = "".join(canonical_json(r) + "\n" for r in records)
    if path.exists() and path.read_text() != body:
        history = path.parent / "history"
        history.mkdir(exist_ok=True)
        (history / (byte_digest(path.read_bytes()) + ".jsonl")).write_bytes(path.read_bytes())
    path.write_text(body)
    return records


def read_manifest(path: Path) -> tuple[Acquisition, ...]:
    records = tuple(
        Acquisition.model_validate_json(line) for line in path.read_text().splitlines() if line
    )
    if len({r.document.id for r in records}) != len(records):
        raise ValueError("duplicate manifest source")
    return records


def compile_corpus(data_root: Path, embedder: Embedder | None = None) -> dict[str, Any]:
    records = read_manifest(data_root / "manifests/corpus.jsonl")
    chunks: list[Chunk] = []
    statuses: list[dict[str, Any]] = []
    for record in records:
        if record.status != "accepted":
            statuses.append(
                {"document_id": record.document.id, "status": record.status, "reason": record.error}
            )
            continue
        try:
            parsed = parse(record.document, replay_bytes(record, data_root / "raw/corpus"))
            group = chunk_document(parsed)
            chunks.extend(group)
            statuses.append(
                {
                    "document_id": record.document.id,
                    "status": "parsed",
                    "paragraphs": len(parsed.paragraphs),
                    "chunks": len(group),
                }
            )
        except ValueError as exc:
            statuses.append(
                {"document_id": record.document.id, "status": "parse_failed", "reason": str(exc)}
            )
    chunks.sort(key=lambda c: c.id)
    output = data_root / "processed/ingestion"
    output.mkdir(parents=True, exist_ok=True)
    # Full source wording remains in ignored interim storage; committed records contain locators only.
    interim = data_root / "interim"
    interim.mkdir(exist_ok=True)
    with (interim / "chunks.jsonl").open("w") as stream:
        for chunk in chunks:
            stream.write(canonical_json(chunk) + "\n")
    metadata = []
    for chunk in chunks:
        metadata.append(
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "document_sha256": chunk.document_sha256,
                "unit_id": chunk.unit_id,
                "text_sha256": byte_digest(chunk.text.encode()),
                "characters": len(chunk.text),
                "paragraphs": [
                    {
                        "id": p.original.id,
                        "page": p.original.page,
                        "section": p.original.section,
                        "dom_path": p.original.dom_path,
                        "characters": len(p.text),
                        "original_characters": len(p.original.text),
                        "text_sha256": byte_digest(p.text.encode()),
                        "offset_map": [s.model_dump() for s in p.offset_map],
                    }
                    for p in chunk.paragraphs
                ],
            }
        )
    (output / "chunks.jsonl").write_text("".join(canonical_json(row) + "\n" for row in metadata))
    if embedder is not None and chunks:
        vectors = np.stack([embedder.embed(c.text) for c in chunks])
        VectorIndex(tuple(c.id for c in chunks), vectors).save(output / "index")
    report = coverage(records, statuses)
    report.update(
        {
            "schema_version": "ingestion-run/1.0",
            "provenance": build_provenance(data_root),
            "manifest_sha256": byte_digest((data_root / "manifests/corpus.jsonl").read_bytes()),
            "statuses": statuses,
            "chunks": len(chunks),
            "paragraphs": sum(len(c.paragraphs) for c in chunks),
            "index_built": embedder is not None,
            "chunks_sha256": byte_digest((output / "chunks.jsonl").read_bytes()),
        }
    )
    (output / "report.json").write_text(canonical_json(report) + "\n")
    return report


def coverage(records: tuple[Acquisition, ...], statuses: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = {x["document_id"] for x in statuses if x["status"] == "parsed"}
    rows = [
        {
            "id": r.document.id,
            "sha256": r.sha256,
            "country": r.document.country,
            "role": r.document.role,
            "grade": g,
        }
        for r in records
        if r.document.id in parsed
        for g in r.document.grades
    ]
    frame = pl.DataFrame(
        rows,
        schema={
            "id": pl.String,
            "sha256": pl.String,
            "country": pl.String,
            "role": pl.String,
            "grade": pl.Int64,
        },
    )
    grades = []
    for g in range(2, 7):
        foreign = frame.filter(
            (pl.col("grade") == g) & (pl.col("role") == "foreign_curriculum")
        ).unique("sha256")
        ids = sorted(foreign["id"].to_list())
        countries = sorted(foreign["country"].unique().to_list())
        grades.append(
            {
                "grade": g,
                "foreign_source_ids": ids,
                "foreign_sources": len(ids),
                "countries": countries,
                "quota_met": 8 <= len(ids) <= 10 and len(countries) >= 4,
            }
        )
    required = [r.document.id for r in records if r.document.required_baseline]
    missing = sorted(set(required) - parsed)
    frameworks = sorted(
        r.document.id
        for r in records
        if r.document.role == "international_framework" and r.document.id in parsed
    )
    return {
        "grades": grades,
        "required_baseline_ids": required,
        "missing_baseline_ids": missing,
        "baseline_complete": bool(required) and not missing,
        "coverage_complete": all(g["quota_met"] for g in grades)
        and bool(required)
        and not missing
        and bool(frameworks),
        "international_frameworks": sorted(
            {
                r.document.id
                for r in records
                if r.document.role == "international_framework" and r.document.id in parsed
            }
        ),
    }
