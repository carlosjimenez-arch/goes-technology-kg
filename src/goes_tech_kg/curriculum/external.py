"""Resolve external prerequisite IDs against an explicit hash-pinned export."""

import json
from pathlib import Path
from typing import Any

from goes_tech_kg.schemas.base import byte_digest


def verify_external_dependency(
    path: Path, expected_sha256: str, node_id: str, requested_grade: int
) -> dict[str, Any]:
    body = path.read_bytes()
    if byte_digest(body) != expected_sha256:
        raise ValueError("external snapshot digest mismatch")
    snapshot = json.loads(body)
    nodes = {n["id"]: n for n in snapshot["nodes"]}
    if len(nodes) != len(snapshot["nodes"]):
        raise ValueError("duplicate external node")
    if node_id not in nodes:
        raise ValueError("external prerequisite ID absent from snapshot")
    node = nodes[node_id]
    available = node["available_grade"]
    return {
        "repository": snapshot["repository"],
        "snapshot_sha256": expected_sha256,
        "node_id": node_id,
        "available_grade": available,
        "requested_grade": requested_grade,
        "conflict": available > requested_grade,
        "validation_status": snapshot["validation_status"],
    }
