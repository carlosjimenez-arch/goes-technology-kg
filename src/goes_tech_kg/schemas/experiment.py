"""Contracts for prompt experiments: the plan that is run and the report it produces.

Plans and reports are committed artifacts, so they are closed contracts rather than free-form
dictionaries. Every field an observation can carry is declared here, including the ones that
exist only when a cell fails, so a malformed report fails loudly instead of silently losing a
column in the report tables.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from goes_tech_kg.schemas.base import Contract, Digest, Text
from goes_tech_kg.schemas.llm import ContextRef, GenerationSettings, TokenUsage

JudgeVerdictLabel = Literal["accept", "revise", "reject"]
OutputStatus = Literal["ok", "refused"]


class ExperimentCase(Contract):
    """One decomposition task: a skill-map entry, its grade and the evidence it may cite."""

    id: Text
    grade: Text
    skill_map_entry: Text
    context_refs: tuple[ContextRef, ...] = ()
    # Only for adversarial cases; labelled synthetic and never treated as source evidence.
    synthetic_context: str | None = None
    expect_refusal: bool = False
    paraphrase_of: str | None = None
    directional_pair: str | None = None
    forbidden_terms: tuple[str, ...] = ()
    notes: str = ""

    @property
    def is_valid_task(self) -> bool:
        """True when the case is a genuine decomposition task, not a refusal or injection probe."""
        return not self.expect_refusal and self.synthetic_context is None


class ModelSpec(Contract):
    """A provider model pinned to an endpoint and generation settings."""

    model: Text
    location: Text
    settings: GenerationSettings = GenerationSettings()


class ExperimentPlan(Contract):
    """The full cross product to run: prompts by models by cases by replicates, plus judges."""

    schema_version: Literal["prompt-experiment/1.0"] = "prompt-experiment/1.0"
    name: Text
    prompt_ids: tuple[Text, ...] = Field(min_length=1)
    models: tuple[ModelSpec, ...] = Field(min_length=1)
    judges: tuple[ModelSpec, ...] = ()
    cases: tuple[ExperimentCase, ...] = Field(min_length=1)
    replicates: int = Field(default=1, ge=1)
    judge_prompt_id: Text = "curricular-judge"
    document_slugs: dict[str, str] = {}

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        ids = [c.id for c in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case id in plan")
        return self

    def case(self, case_id: str) -> ExperimentCase:
        """Look up a case by id, failing loudly when a report names an unplanned case."""
        for candidate in self.cases:
            if candidate.id == case_id:
                return candidate
        raise KeyError(f"case {case_id} is not in plan {self.name}")


class DecompositionMetrics(Contract):
    """Model-free measurements of one decomposition against the context it was given.

    Every value is None when the decomposition or the context cannot support it, never zero:
    a refusal has no quote exactness, and a context without numbered indicators has no coverage.
    """

    micro_skill_count: int = Field(ge=0)
    quote_exactness: float | None = Field(default=None, ge=0, le=1)
    quote_support: float | None = Field(default=None, ge=0, le=1)
    locator_exactness: float | None = Field(default=None, ge=0, le=1)
    observable_rate: float | None = Field(default=None, ge=0, le=1)
    indicator_coverage: float | None = Field(default=None, ge=0, le=1)
    # Coverage over Technology indicators only (decision 0014); None when no scope applies.
    scoped_indicator_coverage: float | None = Field(default=None, ge=0, le=1)
    t0_share: float | None = Field(default=None, ge=0, le=1)
    duplicate_rate: float | None = Field(default=None, ge=0, le=1)
    mean_cognitive_level: float | None = Field(default=None, ge=0, le=2)
    prerequisite_count: int = Field(default=0, ge=0)
    mean_minutes: float | None = Field(default=None, gt=0)


class JudgeAssessment(Contract):
    """One judge's review of one proposal, or the reason the review could not be obtained."""

    judge_model: Text
    self_judged: bool
    request_sha256: Digest
    verdict: JudgeVerdictLabel | None = None
    score: float | None = Field(default=None, ge=0, le=1)
    issue_codes: tuple[str, ...] = ()
    rationale: str | None = None
    usage: TokenUsage | None = None
    transport_error: str | None = None
    parse_error: str | None = None

    @property
    def usable(self) -> bool:
        """True when this assessment carries a verdict that may enter an aggregate."""
        return self.verdict is not None and self.score is not None

    @model_validator(mode="after")
    def verdict_or_reason(self) -> Self:
        if self.verdict is None and not (self.transport_error or self.parse_error):
            raise ValueError("a judgement without a verdict must record why")
        if self.verdict is not None and self.score is None:
            raise ValueError("a verdict requires a score")
        return self


class Observation(Contract):
    """One experiment cell: a prompt, a model, a case and a replicate, with what came back."""

    prompt_id: Text
    prompt_version: Text
    model: Text
    case_id: Text
    replicate: int = Field(ge=0)
    request_sha256: Digest
    parse_ok: bool
    transport_error: str | None = None
    response_sha256: Digest | None = None
    provider_model_version: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    latency_ms: int | None = Field(default=None, ge=0)
    parse_error: str | None = None
    status: OutputStatus | None = None
    refusal_correct: bool | None = None
    metrics: DecompositionMetrics | None = None
    forbidden_term_hits: int | None = Field(default=None, ge=0)
    judgements: tuple[JudgeAssessment, ...] = ()

    @model_validator(mode="after")
    def failure_is_explicit(self) -> Self:
        if self.transport_error is not None:
            if self.parse_ok or self.response_sha256 is not None:
                raise ValueError("a transport failure cannot also carry a parsed response")
            return self
        if self.response_sha256 is None:
            raise ValueError("a cell without a transport error must record its response digest")
        if self.parse_ok and (self.status is None or self.metrics is None):
            raise ValueError("a parsed cell carries both a status and its metrics")
        if not self.parse_ok and self.parse_error is None:
            raise ValueError("an unparsed response must record the validation error")
        return self

    @property
    def judge_scores(self) -> tuple[float, ...]:
        """Scores of judges that are neither self-judging nor failed."""
        return tuple(
            j.score
            for j in self.judgements
            if j.usable and not j.self_judged and j.score is not None
        )


class ExperimentReport(Contract):
    """Everything one run produced: the plan, the context digests and every observation."""

    schema_version: Literal["prompt-experiment-report/1.0"] = "prompt-experiment-report/1.0"
    plan_sha256: Digest
    plan: ExperimentPlan
    context_sha256: dict[str, Digest]
    observations: tuple[Observation, ...]
    acquired: int = Field(ge=0)
    replayed: int = Field(ge=0)

    @model_validator(mode="after")
    def observations_belong_to_plan(self) -> Self:
        planned = {c.id for c in self.plan.cases}
        unknown = {o.case_id for o in self.observations} - planned
        if unknown:
            raise ValueError(f"report contains unplanned cases: {sorted(unknown)}")
        return self
