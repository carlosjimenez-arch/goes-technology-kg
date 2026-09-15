"""LLM output contracts for micro-skill decomposition and judge review."""

import re
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from goes_tech_kg.schemas.base import Contract, Text
from goes_tech_kg.schemas.skills import CognitiveDomain, CTDimension, HalfLife, Strand, Tier

SLUG = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


class EvidenceQuote(Contract):
    quote: Annotated[str, Field(min_length=8)]
    locator_hint: Text


class ProposedMicroSkill(Contract):
    slug: Text
    statement: Text
    observable_verb: Text
    knowledge_object: Text
    strand: Strand
    cognitive_domain: CognitiveDomain
    ct_dimension: CTDimension | None = None
    min_tier: Tier
    t0_alternative: str | None = None
    half_life: HalfLife
    teacher_prep_level: Annotated[int, Field(ge=0, le=3)]
    evidence_of_mastery: Text
    estimated_minutes: Annotated[int, Field(gt=0)]
    prerequisites: tuple[str, ...] = ()
    evidence_quotes: Annotated[tuple[EvidenceQuote, ...], Field(min_length=1)]
    misconceptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def rules(self) -> Self:
        if not SLUG.fullmatch(self.slug):
            raise ValueError("slug must be lowercase kebab-case")
        if self.strand == Strand.COMPUTATIONAL_THINKING and self.ct_dimension is None:
            raise ValueError("computational thinking requires ct_dimension")
        if self.min_tier != Tier.T0 and not self.t0_alternative:
            raise ValueError("a micro-skill above T0 must state its T0 alternative or its absence")
        if self.half_life == HalfLife.VOLATILE:
            raise ValueError("VOLATILE tool operations do not belong in the graph")
        return self


class DecompositionOutput(Contract):
    schema_version: Literal["decomposition/1.0"] = "decomposition/1.0"
    status: Literal["ok", "refused"]
    refusal_reason: str | None = None
    # Defaults are kept so v1 replay keys stay stable; prompt rule 15 demands both fields.
    micro_skills: tuple[ProposedMicroSkill, ...] = ()
    coverage_notes: str = ""

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.status == "refused":
            if self.micro_skills or not self.refusal_reason:
                raise ValueError("refusal carries a reason and no micro-skills")
            return self
        if not self.micro_skills:
            raise ValueError("ok status requires at least one micro-skill")
        slugs = [m.slug for m in self.micro_skills]
        if len(set(slugs)) != len(slugs):
            raise ValueError("duplicate slug")
        for m in self.micro_skills:
            if m.slug in m.prerequisites or any(p not in slugs for p in m.prerequisites):
                raise ValueError("prerequisites must reference other proposed slugs")
        return self


class JudgeIssue(Contract):
    code: Literal[
        "grade_mismatch",
        "not_observable",
        "quote_not_supporting",
        "quote_not_in_context",
        "tier_wrong",
        "duplicate",
        "prerequisite_unjustified",
        "scope_creep",
        "missing_indicator",
        "wrong_strand",
        "wrong_cognitive_domain",
        "minutes_implausible",
        "other",
    ]
    micro_skill_slug: str | None = None
    detail: Text


class JudgeOutput(Contract):
    schema_version: Literal["judge-review/1.0"] = "judge-review/1.0"
    verdict: Literal["accept", "revise", "reject"]
    score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    issues: tuple[JudgeIssue, ...] = ()
    rationale: Text

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.verdict == "accept" and self.score < 0.8:
            raise ValueError("accept requires score >= 0.8")
        if self.verdict == "reject" and self.score >= 0.5:
            raise ValueError("reject requires score < 0.5")
        return self
