"""Validate architecture choices, including the retained Phase 1 evidence extensions."""

from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from goes_tech_kg.schemas.base import Contract, Text


class DecisionOption(Contract):
    model_config = ConfigDict(extra="allow", frozen=True)
    id: Text
    name: Text


class EnforcementTest(Contract):
    model_config = ConfigDict(extra="allow", frozen=True)
    id: Text
    assertion: Text


class Enforcement(Contract):
    status: Literal["specified_not_implemented", "implemented"]
    tests: list[EnforcementTest] = Field(min_length=1)


class ArchitectureDecision(Contract):
    # Phase 1 records carry research matrices and citations beyond this common contract.
    model_config = ConfigDict(extra="allow", frozen=True)
    schema_version: Text
    id: Text
    title: Text
    status: Literal["proposed", "pending", "accepted", "superseded"]
    options: list[DecisionOption] = Field(min_length=2)
    criteria: list[Any] = Field(min_length=1)
    selected_option: Text
    rationale: list[Text] = Field(min_length=1)
    enforced_by: Enforcement | None = None

    @model_validator(mode="after")
    def choice(self) -> Self:
        ids = [option.id for option in self.options]
        if len(ids) != len(set(ids)) or self.selected_option not in ids:
            raise ValueError("decision must select exactly one declared option")
        return self
