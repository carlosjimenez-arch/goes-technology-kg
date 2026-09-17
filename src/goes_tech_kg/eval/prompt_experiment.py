"""Run a prompt experiment: every prompt by model by case cell, recorded and measured.

The harness owns no provider logic and no scoring opinion. It renders the context, asks the
gateway (which replays unless online acquisition is authorized), validates the answer against
its contract, measures it without a model, and optionally asks each judge. Everything it
returns is a validated contract, so a failed cell is visible rather than a missing key.
"""

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from goes_tech_kg.eval.prompt_metrics import ContextIndex, forbidden_term_hits, summarize
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.prompts.decomposition import DECOMPOSITION_PROMPTS
from goes_tech_kg.prompts.judge import JUDGE_PROMPTS
from goes_tech_kg.prompts.registry import Prompt
from goes_tech_kg.retrieval.context import ChunkLookup, assemble_context
from goes_tech_kg.schemas.base import byte_digest, canonical_json
from goes_tech_kg.schemas.decomposition import DecompositionOutput, JudgeOutput
from goes_tech_kg.schemas.experiment import (
    ExperimentCase,
    ExperimentPlan,
    ExperimentReport,
    JudgeAssessment,
    ModelSpec,
    Observation,
)
from goes_tech_kg.schemas.llm import LLMRequest, ResponseRecord, provider_schema
from goes_tech_kg.schemas.scope import TechnologyScope

#: How much of a provider or validation error is kept in a report cell.
ERROR_EXCERPT = 400


def render_context(case: ExperimentCase, lookup: ChunkLookup, slugs: dict[str, str]) -> str:
    """Assemble the case's evidence into one locator-prefixed block, synthetic parts last."""
    parts = []
    if case.context_refs:
        parts.append(assemble_context(case.context_refs, lookup, slugs))
    if case.synthetic_context is not None:
        parts.append(case.synthetic_context)
    return "\n".join(parts)


def build_request(
    prompt: Prompt,
    spec: ModelSpec,
    case: ExperimentCase,
    context: str,
    replicate: int,
    extra: dict[str, str] | None = None,
    output_schema: dict[str, Any] | None = None,
    case_id: str | None = None,
) -> tuple[LLMRequest, str]:
    """Render the prompt and pin every field that can change the answer into the replay key.

    `case_id` overrides the recorded identity, which the judge path needs so that judging two
    different proposals of the same case yields two different cache keys.
    """
    values = {
        "case_id": case.id,
        "grade": case.grade,
        "skill_map_entry": case.skill_map_entry,
        "context": context,
        "output_schema_version": prompt.output_schema_version,
    }
    values.update(extra or {})
    user_content = prompt.render(**values)
    request = LLMRequest(
        model=spec.model,
        location=spec.location,
        prompt_id=prompt.id,
        prompt_version=prompt.version,
        system_instruction=prompt.system,
        user_content_sha256=byte_digest(user_content.encode()),
        case_id=case_id or case.id,
        context_refs=case.context_refs,
        output_schema_version=prompt.output_schema_version,
        output_json_schema=output_schema or provider_schema(DecompositionOutput),
        settings=spec.settings,
        replicate=replicate,
    )
    return request, user_content


def parse_output(record: ResponseRecord) -> tuple[DecompositionOutput | None, str | None]:
    """Validate a recorded answer against the decomposition contract, keeping the error text."""
    try:
        return DecompositionOutput.model_validate_json(record.response_text), None
    except (ValidationError, ValueError) as exc:
        return None, str(exc)[:ERROR_EXCERPT]


def verification_text(output: DecompositionOutput, context: ContextIndex) -> str:
    """Programmatic quote verification handed to the judge so it does not re-check literalness."""
    total = sum(len(m.evidence_quotes) for m in output.micro_skills)
    missing = context.unsupported_quotes(output)
    lines = [f"Citas verificadas: {total - len(missing)} de {total} sostenidas por el contexto."]
    lines.extend(f"- NO SOSTENIDA en '{slug}': {quote[:200]}" for slug, quote in missing)
    if not missing:
        lines.append("- Todas las citas están sostenidas; no reportés quote_not_in_context.")
    return "\n".join(lines)


def judge_proposal(
    judge_prompt: Prompt,
    judge: ModelSpec,
    proposer: ModelSpec,
    case: ExperimentCase,
    context: ContextIndex,
    proposal: ResponseRecord,
    output: DecompositionOutput,
    gateway: LLMGateway,
) -> JudgeAssessment:
    """Ask one judge to review one proposal; transport and validation failures become fields."""
    extra = {"proposal": proposal.response_text}
    if "verification" in judge_prompt.variables:
        extra["verification"] = verification_text(output, context)
    request, user_content = build_request(
        judge_prompt,
        judge,
        case,
        context.text,
        0,
        extra=extra,
        output_schema=provider_schema(JudgeOutput),
        case_id=f"{case.id}#{proposal.response_sha256[:12]}",
    )
    common = {
        "judge_model": judge.model,
        "self_judged": judge.model == proposer.model,
        "request_sha256": request.key,
    }
    try:
        record = gateway.complete(request, user_content)
    except RuntimeError as exc:
        return JudgeAssessment(**common, transport_error=str(exc)[:ERROR_EXCERPT])
    try:
        verdict = JudgeOutput.model_validate_json(record.response_text)
    except (ValidationError, ValueError) as exc:
        return JudgeAssessment(**common, parse_error=str(exc)[:ERROR_EXCERPT])
    return JudgeAssessment(
        **common,
        verdict=verdict.verdict,
        score=verdict.score,
        issue_codes=tuple(sorted({i.code for i in verdict.issues})),
        rationale=verdict.rationale,
        usage=record.usage,
    )


def observe(
    plan: ExperimentPlan,
    prompt: Prompt,
    spec: ModelSpec,
    case: ExperimentCase,
    context: ContextIndex,
    replicate: int,
    gateway: LLMGateway,
    in_scope: frozenset[str] | None = None,
) -> Observation:
    """Run one cell end to end and return everything measured about it."""
    request, user_content = build_request(prompt, spec, case, context.text, replicate)
    identity = {
        "prompt_id": prompt.id,
        "prompt_version": prompt.version,
        "model": spec.model,
        "case_id": case.id,
        "replicate": replicate,
        "request_sha256": request.key,
    }
    try:
        record = gateway.complete(request, user_content)
    except RuntimeError as exc:
        return Observation(**identity, parse_ok=False, transport_error=str(exc)[:ERROR_EXCERPT])
    output, error = parse_output(record)
    delivery = {
        "response_sha256": record.response_sha256,
        "provider_model_version": record.provider_model_version,
        "finish_reason": record.finish_reason,
        "usage": record.usage,
        "latency_ms": record.latency_ms,
    }
    if output is None:
        return Observation(**identity, **delivery, parse_ok=False, parse_error=error)
    judgements = tuple(
        judge_proposal(
            JUDGE_PROMPTS[plan.judge_prompt_id],
            judge,
            spec,
            case,
            context,
            record,
            output,
            gateway,
        )
        for judge in plan.judges
    )
    return Observation(
        **identity,
        **delivery,
        parse_ok=True,
        status=output.status,
        refusal_correct=(output.status == "refused") == case.expect_refusal,
        metrics=summarize(output, context, in_scope),
        forbidden_term_hits=(
            forbidden_term_hits(output, case.forbidden_terms) if case.forbidden_terms else None
        ),
        judgements=judgements,
    )


def _scoped_indicators(scope: TechnologyScope | None, case_id: str) -> frozenset[str] | None:
    """Technology indicators of the unit a case cites, or None when no scope covers it."""
    if scope is None:
        return None
    unit = scope.for_case(case_id)
    return unit.technology_indicators if unit is not None else None


def run_experiment(
    plan: ExperimentPlan,
    gateway: LLMGateway,
    lookup: ChunkLookup,
    output_dir: Path,
    workers: int = 1,
    scope: TechnologyScope | None = None,
) -> ExperimentReport:
    """Run every cell of the plan, write the report and return it.

    `workers` only parallelizes acquisition; every cell has a distinct replay key, so
    concurrent writes cannot race on one record. `scope` enables Technology-only coverage.
    """
    contexts = {
        c.id: ContextIndex(render_context(c, lookup, plan.document_slugs)) for c in plan.cases
    }
    scopes = {c.id: _scoped_indicators(scope, c.id) for c in plan.cases}
    cells = [
        (DECOMPOSITION_PROMPTS[prompt_id], spec, case, replicate)
        for prompt_id in plan.prompt_ids
        for spec in plan.models
        for case in plan.cases
        for replicate in range(plan.replicates)
    ]

    def run_cell(args: tuple[Prompt, ModelSpec, ExperimentCase, int]) -> Observation:
        prompt, spec, case, replicate = args
        return observe(
            plan, prompt, spec, case, contexts[case.id], replicate, gateway, scopes[case.id]
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        observations = tuple(pool.map(run_cell, cells))
    report = ExperimentReport(
        plan_sha256=byte_digest(canonical_json(plan).encode()),
        plan=plan,
        context_sha256={k: byte_digest(v.text.encode()) for k, v in sorted(contexts.items())},
        observations=observations,
        acquired=gateway.acquired,
        replayed=gateway.replayed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(canonical_json(report) + "\n")
    return report


def load_plan(path: Path) -> ExperimentPlan:
    """Read a committed plan; its context references pin the parser version it ran on."""
    return ExperimentPlan.model_validate(json.loads(path.read_text()))


def load_report(path: Path) -> ExperimentReport:
    """Read a committed report, validating every observation against its contract."""
    return ExperimentReport.model_validate(json.loads(path.read_text()))


def load_outputs(
    report: ExperimentReport, store: Path, observations: Iterable[Observation] | None = None
) -> dict[tuple[str, str, str], DecompositionOutput]:
    """Recover the parsed decompositions of a report from the recorded responses.

    Keyed by (prompt_id, model, case_id). Only cells that parsed into a decomposition appear,
    so callers never have to re-check `parse_ok` or re-validate the JSON themselves.
    """
    outputs: dict[tuple[str, str, str], DecompositionOutput] = {}
    for row in observations if observations is not None else report.observations:
        if not row.parse_ok or row.status != "ok" or row.response_sha256 is None:
            continue
        record = ResponseRecord.model_validate_json(
            (store / f"{row.request_sha256}.json").read_bytes()
        )
        outputs[(row.prompt_id, row.model, row.case_id)] = DecompositionOutput.model_validate_json(
            record.response_text
        )
    return outputs
