from pathlib import Path

import pytest

from goes_tech_kg.eval import prompt_report
from goes_tech_kg.eval.golden_context import golden_lookup
from goes_tech_kg.eval.prompt_experiment import (
    ExperimentCase,
    load_plan,
    render_context,
    run_experiment,
    verification_text,
)
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.llm.replay import ReplayMiss, ReplayStore
from goes_tech_kg.schemas.base import byte_digest
from goes_tech_kg.schemas.decomposition import DecompositionOutput

PLAN = Path("data/processed/prompt_experiments/plan-replay-fixture.json")
STORE = Path("data/processed/llm_responses")


def test_recorded_experiment_replays_offline_and_reports(tmp_path: Path):
    plan = load_plan(PLAN)
    lookup, slugs = golden_lookup(Path("tests/golden"))
    assert plan.document_slugs == slugs
    gateway = LLMGateway(ReplayStore(STORE), online=None)
    report = run_experiment(plan, gateway, lookup, tmp_path)
    assert report["acquired"] == 0 and report["replayed"] >= len(report["observations"])
    assert (tmp_path / "report.json").exists()
    rows = report["observations"]
    assert len(rows) == len(plan.prompt_ids) * len(plan.cases)
    assert all(row["parse_ok"] for row in rows), [row.get("parse_error") for row in rows]
    valid = [row for row in rows if row["case_id"] == "fixture-ct-g3-secuencias"]
    malformed = [row for row in rows if row["case_id"] == "fixture-malformed-grade-9"]
    assert all(row["status"] == "refused" and row["refusal_correct"] for row in malformed)
    for row in valid:
        assert row["status"] == "ok" and row["metrics"]["micro_skill_count"] >= 1
        assert row["metrics"]["quote_support"] is not None
        judged = [j for j in row["judgements"] if "score" in j]
        assert judged and all(0 <= j["score"] <= 1 for j in judged)
    frame = prompt_report.observations_frame(report)
    assert set(prompt_report.eligibility(frame)["prompt_id"]) == set(plan.prompt_ids)
    assert prompt_report.ranking(frame).height == len(plan.prompt_ids)
    assert prompt_report.worst_cells(frame).height == len(valid)
    second = run_experiment(plan, LLMGateway(ReplayStore(STORE)), lookup, tmp_path / "again")
    assert second["observations"] == report["observations"]


def test_unrecorded_request_fails_instead_of_calling_a_model(tmp_path: Path):
    plan = load_plan(PLAN)
    lookup, _ = golden_lookup(Path("tests/golden"))
    changed = plan.model_copy(
        update={
            "cases": (
                ExperimentCase.model_validate(
                    {**plan.cases[0].model_dump(mode="json"), "grade": "4"}
                ),
            )
        }
    )
    with pytest.raises(ReplayMiss):
        run_experiment(changed, LLMGateway(ReplayStore(STORE)), lookup, tmp_path)


def test_context_is_reproducible_and_never_stored_in_records():
    plan = load_plan(PLAN)
    lookup, _ = golden_lookup(Path("tests/golden"))
    context = render_context(plan.cases[0], lookup, plan.document_slugs)
    assert "precise and unambiguous instructions" in context and context.startswith("[")
    store = ReplayStore(STORE)
    records = [store.path(k).read_text() for k in store.keys()]
    # Records may carry short evidence quotes (model output) but never the rendered context.
    assert records and all(context not in r and '"user_content"' not in r for r in records)
    assert all("Authorization" not in r and "Bearer " not in r for r in records)
    output = DecompositionOutput.model_validate(
        {
            "status": "ok",
            "coverage_notes": "",
            "micro_skills": [
                {
                    "slug": "predecir-programa",
                    "statement": "Predice el resultado de un programa simple.",
                    "observable_verb": "Predice",
                    "knowledge_object": "programa simple",
                    "strand": "computational_thinking",
                    "cognitive_domain": "reasoning",
                    "ct_dimension": "algorithms",
                    "min_tier": "T0",
                    "half_life": "DURABLE",
                    "teacher_prep_level": 1,
                    "evidence_of_mastery": "Señala la casilla final.",
                    "estimated_minutes": 45,
                    "evidence_quotes": [
                        {"quote": "predict the behaviour of simple programs", "locator_hint": "x"},
                        {"quote": "cita inventada que no existe", "locator_hint": "y"},
                    ],
                }
            ],
        }
    )
    text = verification_text(output, context)
    assert "1 de 2" in text and "NO SOSTENIDA" in text
    assert byte_digest(context.encode()) == byte_digest(
        render_context(plan.cases[0], lookup, plan.document_slugs).encode()
    )
