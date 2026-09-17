"""Review revised proposals without showing the earlier critique; judgments remain uncalibrated."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from goes_tech_kg.config import Settings
from goes_tech_kg.curriculum.generate import call, checks, versioned_body
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.llm.replay import ReplayStore
from goes_tech_kg.llm.vertex import VertexClient, gcloud_token_provider
from goes_tech_kg.prompts.release import REVIEW
from goes_tech_kg.retrieval.release_context import EvidenceIndex
from goes_tech_kg.schemas.base import canonical_json, digest
from goes_tech_kg.schemas.llm import ContextRef
from goes_tech_kg.schemas.release import TeachingUnit, UnitPlan, UnitReview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    evidence = EvidenceIndex(Path("data"))
    client = (
        VertexClient(str(Settings().gcp_project), gcloud_token_provider()) if args.online else None
    )
    gateway = LLMGateway(ReplayStore(Path("data/processed/llm_responses")), client)
    paths = sorted(Path("data/processed/curriculum_experiment").glob("g*.json"))
    if len(paths) != 25:
        raise ValueError("Expected 25 units before the independent revision audit")

    editorial_path = Path("evaluation/release_editorial_review.yaml")
    editorial = (
        yaml.safe_load(editorial_path.read_text())["units"] if editorial_path.exists() else {}
    )

    def audit(path: Path) -> str:
        row = json.loads(path.read_text())
        if path.stem in editorial and row.get("editorial_review_sha256") != digest(
            editorial[path.stem]
        ):
            return f"{path.stem}: pending editorial correction; audit deferred"
        unit = TeachingUnit.model_validate(row["revised"])
        plan = UnitPlan.model_validate(row["plan"])
        refs = tuple(ContextRef.model_validate(r) for r in row["context_refs"])
        errors = checks(unit, plan, refs, evidence)
        raw, key = call(
            gateway,
            REVIEW,
            UnitReview,
            plan,
            refs,
            "gemini-2.5-pro",
            unit=canonical_json(plan),
            context=evidence.render(refs),
            proposal=canonical_json(unit),
            checks=canonical_json(errors),
        )
        repairs = []
        review = UnitReview.model_validate(
            versioned_body(raw, "unit-review/1.0", "final_audit", repairs)
        )
        row["final_review"] = review.model_dump(mode="json")
        row["selected_version"] = "initial" if errors and not row["initial_errors"] else "revised"
        row["selection_basis"] = (
            "Deterministic evidence validity; model critiques remain advisory and cannot certify adoption."
        )
        row["request_keys"] = sorted(set(row["request_keys"] + [key]))
        row["format_repairs"] = [
            r for r in row.get("format_repairs", []) if r["stage"] != "final_audit"
        ] + repairs
        path.write_text(canonical_json(row) + "\n")
        return f"{plan.id}: audit issues={len(review.issues)}, selected={row['selected_version']}"

    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(audit, paths):
            print(result, flush=True)
    print(f"acquired={gateway.acquired} replayed={gateway.replayed}")


if __name__ == "__main__":
    main()
