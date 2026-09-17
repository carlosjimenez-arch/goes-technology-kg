"""Evidence-backed curriculum release contracts; expert judgments remain proposals."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from goes_tech_kg.schemas.base import Contract, Grade, Text
from goes_tech_kg.schemas.decomposition import ProposedMicroSkill
from goes_tech_kg.schemas.skills import Strand


class UnitPlan(Contract):
    id: Text
    grade: Grade
    strand: Strand
    title: Text
    learning_goal: Text
    progression: Text
    retrieval_query: Text
    source_units: tuple[tuple[str, str], ...]
    order: Annotated[int, Field(ge=1, le=5)]


class TeachingSkill(ProposedMicroSkill):
    pedagogical_rationale: Text
    grade_rationale: Text
    assessment_task: Text
    success_criteria: Annotated[tuple[Text, ...], Field(min_length=2)]
    teaching_sequence: Annotated[tuple[Text, ...], Field(min_length=3)]
    accessibility_support: Text
    required_resources: tuple[Text, ...]
    external_knowledge: tuple[Text, ...] = ()
    evidence_limitations: Text


class TeachingUnit(Contract):
    schema_version: Literal["teaching-unit/1.0"] = "teaching-unit/1.0"
    unit_id: Text
    grade: Grade
    title: Text
    micro_skills: Annotated[tuple[TeachingSkill, ...], Field(min_length=1, max_length=8)]
    rationale: Text
    assessment_minutes: Annotated[int, Field(ge=0, le=120)]
    setup_minutes: Annotated[int, Field(ge=0, le=120)]
    review_minutes: Annotated[int, Field(ge=1, le=120)]
    safety_notes: Text
    limitations: Annotated[tuple[Text, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        slugs = [m.slug for m in self.micro_skills]
        if len(set(slugs)) != len(slugs):
            raise ValueError("duplicate teaching skill")
        for micro in self.micro_skills:
            if any(len(q.quote.split()) > 20 for q in micro.evidence_quotes):
                raise ValueError("citation excerpt exceeds 20 words")
        return self


class ReviewIssue(Contract):
    code: Literal[
        "evidence",
        "grade",
        "scope",
        "assessment",
        "resources",
        "prerequisite",
        "workload",
        "accessibility",
        "safety",
        "progression",
    ]
    severity: Literal["major", "minor"]
    skill_slug: Text | None = None
    finding: Text
    correction: Text


class UnitReview(Contract):
    schema_version: Literal["unit-review/1.0"] = "unit-review/1.0"
    unit_id: Text
    issues: tuple[ReviewIssue, ...]
    strengths: tuple[Text, ...]
    unresolved_evidence: tuple[Text, ...]
    human_validation_required: Literal[True] = True


class ReleaseScenario(Contract):
    schema_version: Literal["release-scenario/1.0"] = "release-scenario/1.0"
    minutes_per_grade: Annotated[int, Field(gt=0)]
    mode: Literal["standalone", "transversal", "hybrid"]
    own_share: Annotated[float, Field(ge=0, le=1)] = 1.0
    host_capacities: dict[str, int] = {}
    host_indicator: Text | None = None
    available_resources: tuple[Text, ...] | None = None
    material_tier: Literal["T0", "T1", "T2", "T3"] = "T3"
    official_verified: Literal[False] = False

    @model_validator(mode="after")
    def budget(self) -> Self:
        if self.mode == "standalone" and self.own_share != 1:
            raise ValueError("standalone needs own capacity")
        if self.mode == "transversal" and self.own_share != 0:
            raise ValueError("transversal cannot use own capacity")
        if self.mode != "standalone" and (not self.host_capacities or not self.host_indicator):
            raise ValueError("borrowing requires explicit host capacity and indicator")
        if any(v < 0 for v in self.host_capacities.values()):
            raise ValueError("negative host capacity")
        return self


class RemovedPrerequisite(Contract):
    unit_id: Text
    skill_slug: Text
    prerequisite_slug: Text
    rationale: Text


class CurriculumCuration(Contract):
    schema_version: Literal["curriculum-curation/1.0"] = "curriculum-curation/1.0"
    status: Literal["agent_authored_not_human_validated"] = "agent_authored_not_human_validated"
    removed_prerequisites: tuple[RemovedPrerequisite, ...] = ()
