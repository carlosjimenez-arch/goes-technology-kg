"""Golden curriculum records: reference decompositions and prerequisite edges for evaluation."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from goes_tech_kg.schemas.base import Contract, Grade, Text
from goes_tech_kg.schemas.skills import CognitiveDomain, Strand, Tier

AnnotationMethod = Literal["human_unaided", "llm_expert_cross_vendor"]


class GoldenMicroSkill(Contract):
    id: Text
    statement: Text
    grade: Grade
    strand: Strand
    cognitive_domain: CognitiveDomain
    min_tier: Tier
    # Evidence keys align system output to this item: indicator ids ("5.3") for the official
    # programme, paragraph keys ("p94") for foreign sources. At least one is required.
    evidence_keys: tuple[Text, ...] = Field(min_length=1)
    notes: str = ""


class GoldenEdge(Contract):
    source: Text
    target: Text
    type: Literal["PREREQUISITE", "CO_REQUISITE"] = "PREREQUISITE"
    justification: Text


class GoldenRecord(Contract):
    schema_version: Literal["golden-record/1.0"] = "golden-record/1.0"
    case_id: Text
    grade: Grade
    skill_map_entry: Text
    baseline_version: Text
    annotation_method: AnnotationMethod
    annotators: tuple[Text, ...] = Field(min_length=1)
    status: Literal["provisional", "reviewed", "disputed"]
    adjudication: str = ""
    micro_skills: tuple[GoldenMicroSkill, ...] = Field(min_length=1)
    edges: tuple[GoldenEdge, ...] = ()

    @model_validator(mode="after")
    def integrity(self) -> Self:
        ids = [m.id for m in self.micro_skills]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate golden micro-skill id")
        known = set(ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known or edge.source == edge.target:
                raise ValueError("golden edge must join two distinct golden micro-skills")
        pairs = [(e.source, e.target) for e in self.edges]
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate golden edge")
        if self.annotation_method == "human_unaided" and self.status == "provisional":
            raise ValueError("human_unaided records are reviewed or disputed, never provisional")
        return self

    @property
    def edge_set(self) -> frozenset[tuple[str, str]]:
        return frozenset((e.source, e.target) for e in self.edges if e.type == "PREREQUISITE")


def load_golden(directory: Path, require_human: bool = False) -> tuple[GoldenRecord, ...]:
    """Load golden records; with require_human, anything not human_unaided is rejected."""
    records = []
    for path in sorted(directory.glob("*.yaml")):
        body = yaml.safe_load(path.read_text())
        if body.get("schema_version") != "golden-record/1.0":
            continue
        record = GoldenRecord.model_validate(body)
        if require_human and record.annotation_method != "human_unaided":
            raise ValueError(
                f"{path.name}: annotation_method {record.annotation_method} cannot gate;"
                " human_unaided is required"
            )
        records.append(record)
    if not records:
        raise ValueError(f"no golden records under {directory}")
    ids = [r.case_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate golden case id")
    return tuple(records)
