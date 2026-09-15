"""Prompt x model x case experiment with recorded responses and model-free metrics."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError

from goes_tech_kg.eval import prompt_metrics
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.prompts.decomposition import DECOMPOSITION_PROMPTS
from goes_tech_kg.prompts.judge import JUDGE_PROMPTS
from goes_tech_kg.prompts.registry import Prompt
from goes_tech_kg.retrieval.context import ChunkLookup, assemble_context
from goes_tech_kg.schemas.base import Contract, Text, byte_digest, canonical_json
from goes_tech_kg.schemas.decomposition import DecompositionOutput, JudgeOutput
from goes_tech_kg.schemas.llm import ContextRef, GenerationSettings, LLMRequest, ResponseRecord


class ExperimentCase(Contract):
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


class ModelSpec(Contract):
    model: Text
    location: Text
    settings: GenerationSettings = GenerationSettings()


class ExperimentPlan(Contract):
    schema_version: Literal["prompt-experiment/1.0"] = "prompt-experiment/1.0"
    name: Text
    prompt_ids: tuple[Text, ...] = Field(min_length=1)
    models: tuple[ModelSpec, ...] = Field(min_length=1)
    judges: tuple[ModelSpec, ...] = ()
    cases: tuple[ExperimentCase, ...] = Field(min_length=1)
    replicates: int = Field(default=1, ge=1)
    judge_prompt_id: Text = "curricular-judge"
    document_slugs: dict[str, str] = {}


def render_context(case: ExperimentCase, lookup: ChunkLookup, slugs: dict[str, str]) -> str:
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
) -> tuple[LLMRequest, str]:
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
        case_id=case.id,
        context_refs=case.context_refs,
        output_schema_version=prompt.output_schema_version,
        output_json_schema=output_schema or DecompositionOutput.model_json_schema(),
        settings=spec.settings,
        replicate=replicate,
    )
    return request, user_content


def parse_output(record: ResponseRecord) -> tuple[DecompositionOutput | None, str | None]:
    try:
        output = DecompositionOutput.model_validate_json(record.response_text)
    except (ValidationError, ValueError) as exc:
        return None, str(exc)[:400]
    return output, None


def observe(
    plan: ExperimentPlan,
    prompt: Prompt,
    spec: ModelSpec,
    case: ExperimentCase,
    context: str,
    replicate: int,
    gateway: LLMGateway,
) -> dict[str, Any]:
    request, user_content = build_request(prompt, spec, case, context, replicate)
    row: dict[str, Any] = {
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
        row.update({"transport_error": str(exc)[:300], "parse_ok": False})
        return row
    output, error = parse_output(record)
    row.update(
        {
            "response_sha256": record.response_sha256,
            "provider_model_version": record.provider_model_version,
            "finish_reason": record.finish_reason,
            "usage": record.usage.model_dump(),
            "latency_ms": record.latency_ms,
            "parse_ok": output is not None,
            "parse_error": error,
        }
    )
    if output is None:
        return row
    row["status"] = output.status
    row["refusal_correct"] = (output.status == "refused") == case.expect_refusal
    row["metrics"] = prompt_metrics.summarize(output, context)
    if case.forbidden_terms:
        row["forbidden_term_hits"] = prompt_metrics.forbidden_term_hits(
            output, case.forbidden_terms
        )
    for judge in plan.judges:
        row.setdefault("judgements", []).append(
            judge_output(plan, judge, spec, case, context, record, output, gateway)
        )
    return row


def verification_text(output: DecompositionOutput, context: str) -> str:
    total = sum(len(m.evidence_quotes) for m in output.micro_skills)
    missing = prompt_metrics.unsupported_quotes(output, context)
    lines = [f"Citas verificadas: {total - len(missing)} de {total} sostenidas por el contexto."]
    for slug, quote in missing:
        lines.append(f"- NO SOSTENIDA en '{slug}': {quote[:200]}")
    if not missing:
        lines.append("- Todas las citas están sostenidas; no reportés quote_not_in_context.")
    return "\n".join(lines)


def judge_output(
    plan: ExperimentPlan,
    judge: ModelSpec,
    proposer: ModelSpec,
    case: ExperimentCase,
    context: str,
    proposal: ResponseRecord,
    output: DecompositionOutput,
    gateway: LLMGateway,
) -> dict[str, Any]:
    prompt = JUDGE_PROMPTS[plan.judge_prompt_id]
    extra = {"proposal": proposal.response_text}
    if "verification" in prompt.variables:
        extra["verification"] = verification_text(output, context)
    request, user_content = build_request(
        prompt,
        judge,
        case,
        context,
        0,
        extra=extra,
        output_schema=JudgeOutput.model_json_schema(),
    )
    # The judge key must depend on the proposal being judged, not only on the case.
    request = request.model_copy(update={"case_id": f"{case.id}#{proposal.response_sha256[:12]}"})
    request = LLMRequest.model_validate(request.model_dump(mode="json"))
    result: dict[str, Any] = {
        "judge_model": judge.model,
        "self_judged": judge.model == proposer.model,
        "request_sha256": request.key,
    }
    try:
        record = gateway.complete(request, user_content)
        verdict = JudgeOutput.model_validate_json(record.response_text)
    except RuntimeError as exc:
        result["transport_error"] = str(exc)[:300]
        return result
    except (ValidationError, ValueError) as exc:
        result["parse_error"] = str(exc)[:300]
        return result
    result.update(
        {
            "verdict": verdict.verdict,
            "score": verdict.score,
            "issue_codes": sorted({i.code for i in verdict.issues}),
            "rationale": verdict.rationale,
            "usage": record.usage.model_dump(),
        }
    )
    return result


def run_experiment(
    plan: ExperimentPlan,
    gateway: LLMGateway,
    lookup: ChunkLookup,
    output_dir: Path,
    workers: int = 1,
) -> dict[str, Any]:
    contexts = {c.id: render_context(c, lookup, plan.document_slugs) for c in plan.cases}
    cells = [
        (DECOMPOSITION_PROMPTS[prompt_id], spec, case, replicate)
        for prompt_id in plan.prompt_ids
        for spec in plan.models
        for case in plan.cases
        for replicate in range(plan.replicates)
    ]

    def cell(args: tuple[Prompt, ModelSpec, ExperimentCase, int]) -> dict[str, Any]:
        prompt, spec, case, replicate = args
        return observe(plan, prompt, spec, case, contexts[case.id], replicate, gateway)

    # Every cell has a distinct request key, so concurrent acquisition cannot race on a record.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(cell, cells))
    report = {
        "schema_version": "prompt-experiment-report/1.0",
        "plan_sha256": byte_digest(canonical_json(plan).encode()),
        "plan": plan.model_dump(mode="json"),
        "context_sha256": {k: byte_digest(v.encode()) for k, v in sorted(contexts.items())},
        "observations": rows,
        "acquired": gateway.acquired,
        "replayed": gateway.replayed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(canonical_json(report) + "\n")
    return report


def load_plan(path: Path) -> ExperimentPlan:
    return ExperimentPlan.model_validate(json.loads(path.read_text()))
