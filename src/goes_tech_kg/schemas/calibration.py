"""Confidence is derived from recorded evidence and a golden-fitted calibration, never authored."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from goes_tech_kg.schemas.base import Contract, Digest, Probability, Text

Verdict = Literal["accept", "revise", "reject"]
VERDICT_VALUE: dict[str, float] = {"accept": 1.0, "revise": 0.5, "reject": 0.0}

# A judge below the advisory floor carries no weight; weight grows linearly up to full agreement.
JUDGE_ADVISORY_FLOOR = 0.4


class JudgeVerdict(Contract):
    schema_version: Literal["judge-verdict/1.0"] = "judge-verdict/1.0"
    judge_id: Text
    judge_llm_version: Text
    verdict: Verdict
    # Committed agreement statistic (kappa or alpha) from evaluation/judges/<judge_id>/status.json.
    # None means the judge has not been calibrated against humans and carries no weight.
    agreement_with_humans: Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)] | None
    response_sha256: Digest

    @property
    def weight(self) -> float:
        if self.agreement_with_humans is None or self.agreement_with_humans <= JUDGE_ADVISORY_FLOOR:
            return 0.0
        return (self.agreement_with_humans - JUDGE_ADVISORY_FLOOR) / (1.0 - JUDGE_ADVISORY_FLOOR)


class ConfidenceBasis(Contract):
    """Every signal that feeds a micro-skill's confidence; all of it must be reconstructible."""

    schema_version: Literal["confidence-basis/1.0"] = "confidence-basis/1.0"
    judge_verdicts: tuple[JudgeVerdict, ...] = ()
    citation_support: Probability
    revision_count: Annotated[int, Field(strict=True, ge=0)]
    human_review: Literal["none", "accepted", "corrected", "rejected"] = "none"

    @model_validator(mode="after")
    def canonical(self) -> Self:
        keys = [(v.judge_id, v.response_sha256) for v in self.judge_verdicts]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate judge verdict")
        object.__setattr__(
            self,
            "judge_verdicts",
            tuple(sorted(self.judge_verdicts, key=lambda v: (v.judge_id, v.response_sha256))),
        )
        return self


class CalibrationSample(Contract):
    """One golden item: the raw evidence score the system produced and the unaided human label."""

    raw_score: Probability
    human_accepted: bool
    annotation_method: Literal["human_unaided"]


class CalibrationBin(Contract):
    raw_upper: Probability
    probability: Probability
    support: Annotated[int, Field(strict=True, ge=1)]


class CalibrationTable(Contract):
    """Monotone map from raw evidence score to empirical acceptance probability."""

    schema_version: Literal["calibration-table/1.0"] = "calibration-table/1.0"
    golden_sha256: Digest
    method: Literal["pool-adjacent-violators"] = "pool-adjacent-violators"
    bins: Annotated[tuple[CalibrationBin, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def monotone(self) -> Self:
        uppers = [b.raw_upper for b in self.bins]
        probabilities = [b.probability for b in self.bins]
        if uppers != sorted(uppers) or len(set(uppers)) != len(uppers):
            raise ValueError("bins must have strictly increasing upper bounds")
        if uppers[-1] != 1.0:
            raise ValueError("last bin must cover raw score 1.0")
        if probabilities != sorted(probabilities):
            raise ValueError("calibrated probabilities must be nondecreasing")
        return self

    def apply(self, raw_score: float) -> float:
        if not 0.0 <= raw_score <= 1.0:
            raise ValueError("raw score outside [0, 1]")
        for bin_ in self.bins:
            if raw_score <= bin_.raw_upper:
                return bin_.probability
        raise AssertionError("unreachable: last bin covers 1.0")
