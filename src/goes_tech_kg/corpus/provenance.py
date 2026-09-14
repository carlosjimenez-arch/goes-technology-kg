"""Stable build inputs; wall-clock acquisition times belong only to pinned manifests."""

import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

from goes_tech_kg.schemas.base import byte_digest, digest


def build_provenance(data_root: Path, seed: int = 0) -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    root = package.parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    commit = result.stdout.strip() if result.returncode == 0 else None
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False
    )
    dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    code = {
        str(p.relative_to(package)): byte_digest(p.read_bytes())
        for p in sorted(package.rglob("*.py"))
    }
    model = data_root / "processed/embeddings/model.json"
    lock = root / "uv.lock"
    return {
        "commit": commit,
        "git_dirty": dirty,
        "source_tree_sha256": digest(code),
        "lock_sha256": byte_digest(lock.read_bytes()) if lock.exists() else None,
        "model_manifest_sha256": byte_digest(model.read_bytes()) if model.exists() else None,
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "dependencies": {
            name: version(name)
            for name in ("pydantic", "pypdf", "beautifulsoup4", "numpy", "polars", "scipy")
        },
        "normalizer_version": "crlf-1",
        "parser_version": "reviewed-units-1",
        "llm_calls": 0,
    }
