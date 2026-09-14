"""Context enters only curriculum placement, never a graph archetype."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import model_validator

from goes_tech_kg.schemas.base import Contract, Grade, Minutes, Text, stable_id


class ContextType(StrEnum):
    EVERYDAY = "everyday"
    DISCIPLINARY = "disciplinary"
    INTERDISCIPLINARY = "interdisciplinary"


class ContextDomain(StrEnum):
    POLITICAL = "political"
    CULTURAL = "cultural"
    ECONOMIC = "economic"
    ENVIRONMENTAL = "environmental"


class ContextTag(Contract):
    schema_version: Literal["context-tag/1.0"] = "context-tag/1.0"
    type: ContextType
    domain: ContextDomain


class CurriculumUnit(Contract):
    id: str = ""
    slug: Text
    grade: Grade
    label: Text
    micro_skill_ids: tuple[Text, ...]
    contexts: tuple[ContextTag, ...] = ()
    instruction_minutes: Minutes
    assessment_minutes: Minutes
    setup_minutes: Minutes
    review_minutes: Minutes
    total_minutes: Minutes

    @model_validator(mode="after")
    def totals(self) -> Self:
        if not self.micro_skill_ids or len(set(self.micro_skill_ids)) != len(self.micro_skill_ids):
            raise ValueError("unit needs distinct micro-skills")
        if (
            self.total_minutes
            != self.instruction_minutes
            + self.assessment_minutes
            + self.setup_minutes
            + self.review_minutes
        ):
            raise ValueError("inconsistent unit minutes")
        expected = stable_id(
            "unit",
            self.slug,
            {
                "grade": self.grade,
                "label": self.label,
                "micro_skill_ids": sorted(self.micro_skill_ids),
            },
        )
        if self.id and self.id != expected:
            raise ValueError("unit identity mismatch")
        object.__setattr__(self, "id", expected)
        object.__setattr__(self, "micro_skill_ids", tuple(sorted(self.micro_skill_ids)))
        return self


class BudgetPolicy(Contract):
    mode: Literal["standalone", "transversal", "hybrid"]
    grade: Grade
    own_minutes: Minutes
    borrowed_minutes_cap: Minutes
    host_capacities: dict[str, Minutes]
    official_verified: bool = False

    @model_validator(mode="after")
    def mode_limits(self) -> Self:
        if self.mode == "standalone" and (self.borrowed_minutes_cap or self.host_capacities):
            raise ValueError("standalone cannot borrow")
        if self.mode == "transversal" and self.own_minutes:
            raise ValueError("transversal cannot allocate own hours")
        return self


class BudgetCharge(Contract):
    micro_skill_id: Text
    minutes: Minutes
    host_subject: Text | None = None
    host_indicator: Text | None = None

    @model_validator(mode="after")
    def host_trace(self) -> Self:
        if bool(self.host_subject) != bool(self.host_indicator):
            raise ValueError("borrowed minutes require host subject and indicator")
        return self


class BudgetLedger(Contract):
    policy: BudgetPolicy
    charges: tuple[BudgetCharge, ...]

    @model_validator(mode="after")
    def capacity(self) -> Self:
        own = sum(c.minutes for c in self.charges if c.host_subject is None)
        borrowed = sum(c.minutes for c in self.charges if c.host_subject is not None)
        if own > self.policy.own_minutes or borrowed > self.policy.borrowed_minutes_cap:
            raise ValueError("budget capacity exceeded")
        for host in {c.host_subject for c in self.charges if c.host_subject}:
            if (
                host not in self.policy.host_capacities
                or sum(c.minutes for c in self.charges if c.host_subject == host)
                > self.policy.host_capacities[host]
            ):
                raise ValueError("host capacity exceeded")
        return self
