"""Apply a versioned, explicit editorial critique once; preserve the previous revision."""

import argparse
import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from goes_tech_kg.config import Settings
from goes_tech_kg.curriculum.generate import call, checks, versioned_body
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.llm.replay import ReplayStore
from goes_tech_kg.llm.vertex import VertexClient, gcloud_token_provider
from goes_tech_kg.prompts.release import REVISE
from goes_tech_kg.retrieval.release_context import EvidenceIndex
from goes_tech_kg.schemas.base import canonical_json, digest
from goes_tech_kg.schemas.llm import ContextRef
from goes_tech_kg.schemas.release import TeachingUnit, UnitPlan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    editorial = yaml.safe_load(Path("evaluation/release_editorial_review.yaml").read_text())
    evidence = EvidenceIndex(Path("data"))
    gateway = LLMGateway(
        ReplayStore(Path("data/processed/llm_responses")),
        VertexClient(str(Settings().gcp_project), gcloud_token_provider()) if args.online else None,
    )
    for name, critique in editorial["units"].items():
        path = Path("data/processed/curriculum_experiment") / (name + ".json")
        row = json.loads(path.read_text())
        if row.get("editorial_review_sha256") == digest(critique):
            continue
        plan = UnitPlan.model_validate(row["plan"])
        refs = tuple(ContextRef.model_validate(r) for r in row["context_refs"])
        before = TeachingUnit.model_validate(row["revised"])
        raw, key = call(
            gateway,
            REVISE,
            TeachingUnit,
            plan,
            refs,
            "gemini-2.5-flash",
            unit=canonical_json(plan),
            context=evidence.render(refs),
            proposal=canonical_json(before),
            critique=canonical_json(critique),
            checks=canonical_json(checks(before, plan, refs, evidence)),
        )
        repairs = []
        keys = [key]
        try:
            revised = TeachingUnit.model_validate(
                versioned_body(raw, "teaching-unit/1.0", "editorial_revision", repairs)
            )
        except (ValueError, ValidationError) as exc:
            repaired, repair_key = call(
                gateway,
                REVISE,
                TeachingUnit,
                plan,
                refs,
                "gemini-2.5-flash",
                unit=canonical_json(plan),
                context=evidence.render(refs),
                proposal=canonical_json(raw),
                critique=canonical_json(critique),
                checks="Corregí el contrato; usá citas exactas de 5 a 8 palabras. " + str(exc),
            )
            keys.append(repair_key)
            revised = TeachingUnit.model_validate(
                versioned_body(repaired, "teaching-unit/1.0", "editorial_contract_repair", repairs)
            )
        row["revision_before_editorial_review"] = row["revised"]
        row["revised"] = revised.model_dump(mode="json")
        row["final_errors"] = checks(revised, plan, refs, evidence)
        row["editorial_review_sha256"] = digest(critique)
        row["request_keys"] = sorted(set(row["request_keys"] + keys))
        row["format_repairs"].extend(repairs)
        path.write_text(canonical_json(row) + "\n")
        print(name, row["final_errors"], flush=True)


if __name__ == "__main__":
    main()
