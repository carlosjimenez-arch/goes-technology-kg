"""One additional targeted repair for exact citations or unresolved internal IDs."""

import argparse
import json
from pathlib import Path

from goes_tech_kg.config import Settings
from goes_tech_kg.curriculum.generate import call, checks, versioned_body
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.llm.replay import ReplayStore
from goes_tech_kg.llm.vertex import VertexClient, gcloud_token_provider
from goes_tech_kg.prompts.release import REVISE
from goes_tech_kg.retrieval.release_context import EvidenceIndex
from goes_tech_kg.schemas.base import canonical_json
from goes_tech_kg.schemas.llm import ContextRef
from goes_tech_kg.schemas.release import TeachingUnit, UnitPlan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()
    index = EvidenceIndex(Path("data"))
    gateway = LLMGateway(
        ReplayStore(Path("data/processed/llm_responses")),
        VertexClient(str(Settings().gcp_project), gcloud_token_provider()) if args.online else None,
    )
    for path in sorted(Path("data/processed/curriculum_experiment").glob("g*.json")):
        row = json.loads(path.read_text())
        plan = UnitPlan.model_validate(row["plan"])
        refs = tuple(ContextRef.model_validate(r) for r in row["context_refs"])
        unit = TeachingUnit.model_validate(row["revised"])
        errors = checks(unit, plan, refs, index)
        if not errors:
            continue
        fragments = []
        for m in unit.micro_skills:
            for q in m.evidence_quotes:
                words = q.quote.split()
                for size in range(min(8, len(words)), 4, -1):
                    candidates = []
                    for start in range(len(words) - size + 1):
                        phrase = " ".join(words[start : start + size])
                        try:
                            index.resolve_quote(phrase, q.locator_hint, refs)
                        except ValueError:
                            continue
                        candidates.append(phrase)
                    if candidates:
                        fragments.append(
                            {
                                "skill": m.slug,
                                "source_locator": q.locator_hint,
                                "exact_fragments": sorted(set(candidates)),
                                "constraint": "Use only if it supports the statement; otherwise revise the unsupported statement. A substring is not proof of pedagogy.",
                            }
                        )
                        break
        instructions = canonical_json(
            {
                "errors": errors,
                "exact_fragment_candidates": fragments,
                "rule": "Use 5-8 words copied from one contiguous source phrase. Move general prior knowledge to external_knowledge; internal prerequisites reference existing slugs only. Keep evidence limitations explicit.",
            }
        )
        raw, key = call(
            gateway,
            REVISE,
            TeachingUnit,
            plan,
            refs,
            "gemini-2.5-flash",
            unit=canonical_json(plan),
            context=index.render(refs),
            proposal=canonical_json(unit),
            critique=canonical_json(row["review"]),
            checks=instructions,
        )
        repairs = []
        revised = TeachingUnit.model_validate(
            versioned_body(raw, "teaching-unit/1.0", "targeted_evidence_repair", repairs)
        )
        row["revision_before_targeted_repair"] = row["revised"]
        row["revised"] = revised.model_dump(mode="json")
        row["final_errors"] = checks(revised, plan, refs, index)
        row["request_keys"] = sorted(set(row["request_keys"] + [key]))
        row["format_repairs"].extend(repairs)
        row["targeted_repair_guidance"] = json.loads(instructions)
        path.write_text(canonical_json(row) + "\n")
        print(plan.id, row["final_errors"], flush=True)


if __name__ == "__main__":
    main()
