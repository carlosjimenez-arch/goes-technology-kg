from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from goes_tech_kg.schemas.decisions import ArchitectureDecision


def test_all_decisions_validate_and_invalid_choice_is_rejected():
    paths = sorted(Path("decisions").glob("*.yaml"))
    assert len(paths) == 6
    for path in paths:
        body = yaml.safe_load(path.read_text())
        decision = ArchitectureDecision.model_validate(body)
        assert decision.id == path.name[:4]
        body["selected_option"] = "nonexistent"
        with pytest.raises(ValidationError, match="declared option"):
            ArchitectureDecision.model_validate(body)
