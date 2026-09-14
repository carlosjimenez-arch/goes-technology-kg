from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from goes_tech_kg.schemas.decisions import ArchitectureDecision


def test_all_decisions_validate_and_invalid_choice_is_rejected():
    paths = sorted(Path("decisions").glob("*.yaml"))
    assert paths, "decision records are required"
    ids = []
    for path in paths:
        body = yaml.safe_load(path.read_text())
        decision = ArchitectureDecision.model_validate(body)
        assert decision.id == path.name[:4]
        ids.append(decision.id)
        body["selected_option"] = "nonexistent"
        with pytest.raises(ValidationError, match="declared option"):
            ArchitectureDecision.model_validate(body)
    assert len(set(ids)) == len(ids), "decision IDs must be unique"


def test_enforcement_section_is_validated_when_present():
    body = yaml.safe_load(Path("decisions/0010-system-design.yaml").read_text())
    assert ArchitectureDecision.model_validate(body).enforced_by is not None
    body["enforced_by"] = {"status": "passing", "tests": []}
    with pytest.raises(ValidationError):
        ArchitectureDecision.model_validate(body)
