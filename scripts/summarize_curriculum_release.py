"""Summarize actual response usage and review findings without claiming human validation."""

import json
from collections import Counter
from pathlib import Path

from goes_tech_kg.schemas.base import canonical_json
from goes_tech_kg.schemas.llm import ResponseRecord


def main() -> None:
    units = [
        json.loads(p.read_text())
        for p in sorted(Path("data/processed/curriculum_experiment").glob("g*.json"))
    ]
    used = {key for unit in units for key in unit["request_keys"]}
    models = {}
    recorded = set()
    for path in sorted(Path("data/processed/llm_responses").glob("*.json")):
        response = ResponseRecord.model_validate_json(path.read_bytes())
        request = response.request
        if not request.prompt_id.startswith("release-") or "golden" in request.case_id:
            continue
        recorded.add(request.key)
        entry = models.setdefault(
            request.model,
            {"responses": 0, "prompt_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0},
        )
        entry["responses"] += 1
        for key, value in response.usage.model_dump().items():
            entry[key] += value
    counts = {}
    for phase in ("review", "final_review"):
        counts[phase] = dict(
            sorted(
                Counter(
                    f"{issue['severity']}:{issue['code']}"
                    for unit in units
                    for issue in unit.get(phase, {}).get("issues", [])
                ).items()
            )
        )
    result = {
        "schema_version": "curriculum-run-summary/1.0",
        "units": len(units),
        "initial_candidate_micro_skills": sum(len(u["initial"]["micro_skills"]) for u in units),
        "revised_candidate_micro_skills": sum(len(u["revised"]["micro_skills"]) for u in units),
        "final_audited_units": sum("final_review" in u for u in units),
        "models": models,
        "all_recorded_release_responses": len(recorded),
        "responses_in_selected_provenance_chains": len(used),
        "other_recorded_attempt_keys": sorted(recorded - used),
        "uncalibrated_critique_issue_counts": counts,
        "interpretation": "Counts describe recorded calls and model findings, not cost, accuracy or human-validated pedagogical improvement. Failed and superseded attempts remain versioned.",
    }
    Path("evaluation/curriculum_run_summary.json").write_text(canonical_json(result) + "\n")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
