"""Deterministic discovery from a reviewed, versioned resource catalogue."""

from pathlib import Path

import yaml

from goes_tech_kg.schemas.corpus import SourceDocument


def discover(path: Path) -> tuple[SourceDocument, ...]:
    data = yaml.safe_load(path.read_text())
    if data.get("schema_version") != "discovery/1.0":
        raise ValueError("unsupported discovery schema")
    sources = tuple(SourceDocument.model_validate(row) for row in data["sources"])
    if len({str(s.url) for s in sources}) != len(sources):
        raise ValueError("duplicate discovery URL")
    return tuple(sorted(sources, key=lambda s: s.id))
