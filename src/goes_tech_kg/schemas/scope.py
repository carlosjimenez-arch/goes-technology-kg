"""Reviewed classification of programme indicators into Technology, Science dependency or excluded.

Decision 0014 fixes the rule; this contract makes it checkable. The coverage ceilings are
derived from the classification rather than written by hand, so a ceiling can never drift away
from the indicators it summarizes.
"""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from goes_tech_kg.schemas.base import Contract, Grade, Text

IndicatorClass = Literal["technology", "science_dependency", "excluded"]


class IndicatorScope(Contract):
    """Why one achievement indicator is or is not a Technology outcome."""

    # Pydantic alias: YAML uses the reserved word "class".
    indicator_class: IndicatorClass = Field(alias="class")
    reason: Text


class UnitScope(Contract):
    """One programme unit with every indicator classified and the cases that cite it."""

    grade: Grade
    unit: int = Field(strict=True, ge=1)
    label: Text
    indicators: dict[str, IndicatorScope] = Field(min_length=1)
    #: Experiment case ids whose evidence is this unit, so reports can scope coverage.
    case_ids: tuple[Text, ...] = ()

    @property
    def key(self) -> str:
        """Grade and unit as they appear in the ceilings table, for example "4/1"."""
        return f"{self.grade}/{self.unit}"

    @property
    def technology_indicators(self) -> frozenset[str]:
        """Indicators that count toward Technology coverage."""
        return frozenset(k for k, v in self.indicators.items() if v.indicator_class == "technology")

    @property
    def ceiling(self) -> float:
        """Highest indicator coverage a faithful decomposition of this unit can reach."""
        return round(len(self.technology_indicators) / len(self.indicators), 3)


class TechnologyScope(Contract):
    """The whole classification for one baseline version."""

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
        cases = [c for u in self.units for c in u.case_ids]
        if len(set(cases)) != len(cases):
            raise ValueError("a case cannot belong to two units")
        return self

    def unit(self, grade: int, unit: int) -> UnitScope:
        """Look up one classified unit, failing loudly when it is not classified."""
        for candidate in self.units:
            if candidate.grade == grade and candidate.unit == unit:
                return candidate
        raise KeyError(f"unit {grade}/{unit} is not classified")

    def for_case(self, case_id: str) -> UnitScope | None:
        """The unit an experiment case draws its evidence from, or None when it draws none."""
        for candidate in self.units:
            if case_id in candidate.case_ids:
                return candidate
        return None


def load_scope(path: Path) -> TechnologyScope:
    """Read and validate the committed classification."""
    return TechnologyScope.model_validate(yaml.safe_load(path.read_text()))
