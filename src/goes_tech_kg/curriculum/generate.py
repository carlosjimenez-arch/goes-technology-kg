"""Bounded model proposal/review/revision; deterministic code validates both versions."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.prompts.registry import Prompt
from goes_tech_kg.prompts.release import PROPOSE, REVIEW, REVISE
from goes_tech_kg.retrieval.release_context import EvidenceIndex
from goes_tech_kg.schemas.base import Contract, byte_digest, canonical_json
from goes_tech_kg.schemas.llm import ContextRef, GenerationSettings, LLMRequest, provider_schema
from goes_tech_kg.schemas.release import TeachingUnit, UnitPlan, UnitReview


def checks(
    unit: TeachingUnit, plan: UnitPlan, refs: tuple[ContextRef, ...], evidence: EvidenceIndex
) -> list[str]:
    errors = []
    if unit.grade != plan.grade or unit.unit_id != plan.id:
        errors.append("Unit identity/grade must match the input plan")
    for micro in unit.micro_skills:
        if micro.strand != plan.strand:
            errors.append(f"{micro.slug}: wrong strand")
        for quote in micro.evidence_quotes:
            try:
                evidence.resolve_quote(quote.quote, quote.locator_hint, refs)
            except ValueError as exc:
                errors.append(f"{micro.slug}: {exc}")
    # Kahn's algorithm catches cycles before the schema-to-graph conversion.
    pending = {m.slug: set(m.prerequisites) for m in unit.micro_skills}
    done: set[str] = set()
    while pending:
        ready = sorted(k for k, v in pending.items() if v <= done)
        if not ready:
            errors.append(
                "cyclic or unresolved prerequisite: "
                + ", ".join(sorted(pending))
                + "; prerequisites must contain ONLY slugs from "
                + ", ".join(sorted(m.slug for m in unit.micro_skills))
                + "; put free-form prior knowledge in external_knowledge, never in prerequisites"
            )
            break
        for key in ready:
            done.add(key)
            del pending[key]
    return errors


def serving_schema(node: Any) -> Any:
    """Remove decoder bounds only; full Pydantic validation remains mandatory locally."""
    if isinstance(node, dict):
        return {
            k: serving_schema(v)
            for k, v in node.items()
            if k
            not in {
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "minItems",
                "maxItems",
                "minLength",
                "maxLength",
                "default",
            }
        }
    if isinstance(node, list):
        return [serving_schema(v) for v in node]
    return node


def call(
    gateway: LLMGateway,
    prompt: Prompt,
    contract: type[Contract],
    plan: UnitPlan,
    refs: tuple[ContextRef, ...],
    model: str,
    **values: str,
) -> tuple[dict[str, Any], str]:
    rendered = prompt.render(**values)
    request = LLMRequest(
        model=model,
        location="us-central1",
        prompt_id=prompt.id,
        prompt_version=prompt.version,
        system_instruction=prompt.system,
        user_content_sha256=byte_digest(rendered.encode()),
        case_id=plan.id,
        context_refs=refs,
        output_schema_version=prompt.output_schema_version,
        output_json_schema=serving_schema(provider_schema(contract)),
        settings=GenerationSettings(max_output_tokens=16384, thinking_budget=1024),
    )
    record = gateway.complete(request, rendered)
    body = record.response_text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(body), request.key


def versioned_body(
    body: dict[str, Any], expected: str, stage: str, repairs: list[dict[str, str]]
) -> dict[str, Any]:
    """Record a narrow transport-version repair; never change generated domain content."""
    if expected == "unit-review/1.0" and body.get("human_validation_required") is False:
        repairs.append(
            {
                "stage": stage,
                "field": "human_validation_required",
                "before": "false",
                "after": "true",
            }
        )
        body = {**body, "human_validation_required": True}
    if expected == "teaching-unit/1.0" and isinstance(body.get("micro_skills"), list):
        replacements = {}
        for micro in body["micro_skills"]:
            before = micro.get("slug", "")
            after = before.lower().replace("_", "-")
            if before != after and re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", after):
                replacements[before] = after
                repairs.append({"stage": stage, "field": "slug", "before": before, "after": after})
        if replacements:
            body = {
                **body,
                "micro_skills": [
                    {
                        **m,
                        "slug": replacements.get(m["slug"], m["slug"]),
                        "prerequisites": [
                            replacements.get(p, p) for p in m.get("prerequisites", [])
                        ],
                    }
                    for m in body["micro_skills"]
                ],
            }
    actual = body.get("schema_version")
    if actual == expected:
        return body
    if actual not in {None, "1.0", "1.0.0", "1.1", "1.1.0", "teaching-unit/1.0", "unit-review/1.0"}:
        raise ValueError(f"unsupported generated schema version: {actual}")
    repairs.append(
        {"stage": stage, "field": "schema_version", "before": str(actual), "after": expected}
    )
    return {**body, "schema_version": expected}


def generate_units(
    plans: tuple[UnitPlan, ...],
    evidence: EvidenceIndex,
    gateway: LLMGateway,
    output: Path,
    workers: int = 3,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if len(plans) > 25:
        raise ValueError("release acquisition is limited to 25 units with bounded stage repairs")

    def generate(plan: UnitPlan) -> str:
        format_repairs: list[dict[str, str]] = []
        refs = evidence.select(plan)
        existing_path = output / (plan.id + ".json")
        if existing_path.exists():
            from goes_tech_kg.curriculum.provenance import verify_record

            existing = json.loads(existing_path.read_text())
            if existing["plan"] != plan.model_dump(mode="json") or existing["context_refs"] != [
                r.model_dump(mode="json") for r in refs
            ]:
                raise ValueError("accepted proposal inputs changed; use a new experiment directory")
            verify_record(existing, gateway.store.root, evidence)
            return f"{plan.id}: resumed verified record"
        context = evidence.render(refs)
        base = {"unit": canonical_json(plan), "context": context}
        raw, first_key = call(
            gateway, PROPOSE, TeachingUnit, plan, refs, "gemini-2.5-flash", **base
        )
        extra_keys: list[str] = []

        def validate_candidate(body: dict[str, Any], stage: str) -> TeachingUnit:
            try:
                return TeachingUnit.model_validate(
                    versioned_body(body, "teaching-unit/1.0", stage, format_repairs)
                )
            except (ValidationError, ValueError) as exc:
                repaired, key = call(
                    gateway,
                    REVISE,
                    TeachingUnit,
                    plan,
                    refs,
                    "gemini-2.5-flash",
                    **base,
                    proposal=canonical_json(body),
                    critique="La salida incumple el contrato; conservá las limitaciones de evidencia.",
                    checks="Corregí estos errores sin inventar citas. Cada habilidad requiere evidencia explícita; reformulá lo no respaldado: "
                    + str(exc),
                )
                extra_keys.append(key)
                return TeachingUnit.model_validate(
                    versioned_body(
                        repaired, "teaching-unit/1.0", stage + "_contract_repair", format_repairs
                    )
                )

        proposal = validate_candidate(raw, "proposal")
        initial_errors = checks(proposal, plan, refs, evidence)
        review_raw, review_key = call(
            gateway,
            REVIEW,
            UnitReview,
            plan,
            refs,
            "gemini-2.5-pro",
            **base,
            proposal=canonical_json(proposal),
            checks=canonical_json(initial_errors),
        )
        review = UnitReview.model_validate(
            versioned_body(review_raw, "unit-review/1.0", "critique", format_repairs)
        )
        revised_raw, revised_key = call(
            gateway,
            REVISE,
            TeachingUnit,
            plan,
            refs,
            "gemini-2.5-flash",
            **base,
            proposal=canonical_json(proposal),
            critique=canonical_json(review),
            checks=canonical_json(initial_errors),
        )
        revised = validate_candidate(revised_raw, "revision")
        final_errors = checks(revised, plan, refs, evidence)
        keys = [first_key, review_key, revised_key]
        if final_errors:
            repaired_raw, repair_key = call(
                gateway,
                REVISE,
                TeachingUnit,
                plan,
                refs,
                "gemini-2.5-flash",
                **base,
                proposal=canonical_json(revised),
                critique=canonical_json(review),
                checks=canonical_json(final_errors),
            )
            revised = validate_candidate(repaired_raw, "repair")
            final_errors = checks(revised, plan, refs, evidence)
            keys.append(repair_key)
        body = {
            "plan": plan.model_dump(mode="json"),
            "context_refs": [r.model_dump(mode="json") for r in refs],
            "initial": proposal.model_dump(mode="json"),
            "review": review.model_dump(mode="json"),
            "revised": revised.model_dump(mode="json"),
            "initial_errors": initial_errors,
            "final_errors": final_errors,
            "request_keys": keys + extra_keys,
            "format_repairs": format_repairs,
        }
        (output / (plan.id + ".json")).write_text(canonical_json(body) + "\n")
        return f"{plan.id}: {len(proposal.micro_skills)} -> {len(revised.micro_skills)} skills, errors {len(initial_errors)} -> {len(final_errors)}"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for message in pool.map(generate, plans):
            print(message, flush=True)
