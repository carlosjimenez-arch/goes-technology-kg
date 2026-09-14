from datetime import UTC, datetime

import pytest
import yaml
from pydantic import ValidationError

from goes_tech_kg.corpus.discover import discover
from goes_tech_kg.corpus.license_gate import license_gate
from goes_tech_kg.corpus.pipeline import read_manifest
from goes_tech_kg.schemas.corpus import SourceDocument

POLICY = {"publishers": [{"host": "www.mined.gob.sv", "official": True, "path_prefixes": ["/"]}]}


def document(**updates):
    data = dict(
        slug="generated-graph",
        title="A curriculum produced by this pipeline",
        url="https://www.mined.gob.sv/docs/generated.pdf",
        publisher="SV curriculum authority",
        country="SV",
        role="supplementary",
        provenance="generated",
        grades=[2],
        alignment_rationale="Would confirm itself",
        format="pdf",
        license=None,
        admission_basis="official_publication",
        license_evidence_url="https://www.mined.gob.sv/docs/generated.pdf",
        license_evidence="Official host",
        reviewed_at=datetime(2026, 9, 14, tzinfo=UTC),
    )
    data.update(updates)
    return SourceDocument(**data)


def test_generated_provenance_is_rejected_before_any_fetch():
    decision = license_gate(document(), POLICY)
    assert decision.status == "rejected" and "Generated" in decision.reason
    assert license_gate(document(provenance="external"), POLICY).status == "accepted"


def test_provenance_is_mandatory_and_closed():
    with pytest.raises(ValidationError, match="provenance"):
        SourceDocument(**{k: v for k, v in document().model_dump().items() if k != "provenance"})
    with pytest.raises(ValidationError):
        document(provenance="unknown")


def test_every_committed_source_declares_external_provenance():
    from pathlib import Path

    catalogue = discover(Path("corpus/sources.yaml"))
    assert catalogue and all(s.provenance == "external" for s in catalogue)
    manifest = read_manifest(Path("data/manifests/corpus.jsonl"))
    assert manifest and all(r.document.provenance == "external" for r in manifest)
    raw = yaml.safe_load(Path("corpus/sources.yaml").read_text())
    assert all("provenance" in row for row in raw["sources"])
