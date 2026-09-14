"""Small explicit domain fixtures; no mocked logic or fabricated network responses."""

import base64
import json
from pathlib import Path

import pytest

from goes_tech_kg.schemas.corpus import SourceDocument
from goes_tech_kg.schemas.skills import EvidenceRef, MicroSkill, Skill


@pytest.fixture
def skill():
    return Skill(slug="algorithm", label="Algorithmic execution")


@pytest.fixture
def evidence():
    return EvidenceRef(
        document_id="source-fixture",
        document_sha256="a" * 64,
        paragraph_id="paragraph-fixture",
        start=0,
        end=8,
    )


@pytest.fixture
def micro_factory(skill, evidence):
    def make(**updates):
        data = dict(
            slug="trace-sequence",
            parent_skill_id=skill.id,
            observable_verb="Trace",
            knowledge_object="Sequential execution",
            strand="computational_thinking",
            cognitive_domain="reasoning",
            ct_dimension="algorithms",
            min_tier="T0",
            downgrade_variant=None,
            half_life="DURABLE",
            teacher_prep_level=1,
            evidence_of_mastery="Explain every state transition on an unseen sequence",
            estimated_minutes=30,
            source_refs=[evidence.model_dump()],
        )
        data.update(updates)
        return MicroSkill.model_validate(data)

    return make


@pytest.fixture
def golden_documents():
    result = []
    for path in sorted(Path("tests/golden").glob("*.json")):
        row = json.loads(path.read_text())
        if "payload_base64" in row:
            result.append(
                (
                    SourceDocument.model_validate(row["document"]),
                    base64.b64decode(row["payload_base64"]),
                    row,
                )
            )
    assert len(result) == 3, "Three real recorded document excerpts are required"
    return result
