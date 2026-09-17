"""Build both approved curriculum snapshots from evidence and recorded model responses."""

from pathlib import Path

from goes_tech_kg.curriculum.publish import build_release
from goes_tech_kg.retrieval.release_context import EvidenceIndex
from goes_tech_kg.schemas.base import canonical_json

if __name__ == "__main__":
    root = Path.cwd()
    manifest = build_release(root, EvidenceIndex(root / "data"))
    print(
        canonical_json(
            {
                "artifacts": len(manifest["artifacts_sha256"]),
                "status": manifest["status"],
                "adoption_ready": manifest["adoption_ready"],
            }
        )
    )
