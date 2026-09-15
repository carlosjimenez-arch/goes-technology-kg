"""Reviewed classification of programme indicators into Technology, Science dependency or excluded."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from goes_tech_kg.schemas.base import Contract, Grade, Text

IndicatorClass = Literal["technology", "science_dependency", "excluded"]


class IndicatorScope(Contract):
    # Pydantic alias: YAML uses the reserved word "class".
    indicator_class: IndicatorClass = Field(alias="class")
    reason: Text


class UnitScope(Contract):
    grade: Grade
    unit: int = Field(strict=True, ge=1)
    label: Text
    indicators: dict[str, IndicatorScope] = Field(min_length=1)

    @property
    def key(self) -> str:
        return f"{self.grade}/{self.unit}"

    @property
    def technology_indicators(self) -> frozenset[str]:
        return frozenset(k for k, v in self.indicators.items() if v.indicator_class == "technology")

    @property
    def ceiling(self) -> float:
        return round(len(self.technology_indicators) / len(self.indicators), 3)


class TechnologyScope(Contract):
    schema_version: Literal["technology-scope/1.0"] = "technology-scope/1.0"
    decision: Text
    baseline_version: Text
    rule: Text
    classes: dict[IndicatorClass, Text]
    units: tuple[UnitScope, ...] = Field(min_length=1)
    coverage_ceilings: dict[str, float]

    @model_validator(mode="after")
    def ceilings_are_derived(self) -> Self:
        keys = [u.key for u in self.units]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate unit")
        expected = {u.key: u.ceiling for u in self.units}
        if self.coverage_ceilings != expected:
            raise ValueError(f"coverage_ceilings must equal derived values {expected}")
        return self

    def unit(self, grade: int, unit: int) -> UnitScope:
        for candidate in self.units:
            if candidate.grade == grade and candidate.unit == unit:
                return candidate
        raise KeyError(f"unit {grade}/{unit} is not classified")


def load_scope(path: Path) -> TechnologyScope:
    return TechnologyScope.model_validate(yaml.safe_load(path.read_text()))
