"""Reject proposals detached from their recorded provider answers."""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from goes_tech_kg.curriculum.generate import checks, versioned_body
from goes_tech_kg.prompts.release import REVIEW
from goes_tech_kg.retrieval.release_context import EvidenceIndex
from goes_tech_kg.schemas.base import byte_digest, canonical_json
from goes_tech_kg.schemas.llm import ContextRef, ResponseRecord
from goes_tech_kg.schemas.release import TeachingUnit, UnitPlan, UnitReview


def verify_record(
    record: dict[str, Any], store: Path, evidence: EvidenceIndex | None = None
) -> None:
    candidates: dict[str, set[str]] = {"teaching-unit/1.0": set(), "unit-review/1.0": set()}
    verified_final = evidence is None or "final_review" not in record
    expected_audit_hash = None
    if evidence is not None and "final_review" in record:
        plan = UnitPlan.model_validate(record["plan"])
        unit = TeachingUnit.model_validate(record["revised"])
        refs = tuple(ContextRef.model_validate(r) for r in record["context_refs"])
        expected_audit_hash = byte_digest(
            REVIEW.render(
                unit=canonical_json(plan),
                context=evidence.render(refs),
                proposal=canonical_json(unit),
                checks=canonical_json(checks(unit, plan, refs, evidence)),
            ).encode()
        )
    for key in record["request_keys"]:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid replay key")
        response = ResponseRecord.model_validate_json((store / (key + ".json")).read_bytes())
        if response.request_sha256 != key or response.request.case_id != record["plan"]["id"]:
            raise ValueError("response belongs to a different unit")
        if [r.model_dump(mode="json") for r in response.request.context_refs] not in [
            record["context_refs"],
            record.get("initial_context_refs", record["context_refs"]),
            *record.get("context_history", []),
        ]:
            raise ValueError("response context differs from proposal context")
        text = response.response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        raw = json.loads(text)
        schema = response.request.output_schema_version
        if schema not in candidates:
            raise ValueError("unexpected release response schema")
        contract = TeachingUnit if schema == "teaching-unit/1.0" else UnitReview
        try:
            parsed = contract.model_validate(versioned_body(raw, schema, "verification", []))
        except (ValidationError, ValueError):
            # Failed candidates are retained as evidence of why a repair was needed.
            continue
        candidates[schema].add(canonical_json(parsed))
        if (
            schema == "unit-review/1.0"
            and "final_review" in record
            and canonical_json(parsed) == canonical_json(record["final_review"])
            and response.request.user_content_sha256 == expected_audit_hash
        ):
            verified_final = True
    for field, schema in [
        ("initial", "teaching-unit/1.0"),
        ("revised", "teaching-unit/1.0"),
        ("review", "unit-review/1.0"),
        ("final_review", "unit-review/1.0"),
    ]:
        if field in record and canonical_json(record[field]) not in candidates[schema]:
            raise ValueError(f"{field} differs from recorded model responses")

    if not verified_final:
        raise ValueError("final review does not evaluate the current revision and evidence")
