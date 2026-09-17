from pathlib import Path

import pytest

from goes_tech_kg.eval import prompt_report
from goes_tech_kg.eval.golden_context import golden_lookup
from goes_tech_kg.eval.prompt_experiment import (
    load_outputs,
    load_plan,
    render_context,
    run_experiment,
    verification_text,
)
from goes_tech_kg.eval.prompt_metrics import ContextIndex
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.llm.replay import ReplayMiss, ReplayStore
from goes_tech_kg.schemas.base import byte_digest
from goes_tech_kg.schemas.decomposition import DecompositionOutput
from goes_tech_kg.schemas.experiment import ExperimentCase

PLAN = Path("data/processed/prompt_experiments/plan-replay-fixture.json")
STORE = Path("data/processed/llm_responses")
GOLDEN = Path("tests/golden")


def test_recorded_experiment_replays_offline_and_reports(tmp_path: Path):
    plan = load_plan(PLAN)
    lookup, slugs = golden_lookup(GOLDEN)
    assert plan.document_slugs == slugs
    gateway = LLMGateway(ReplayStore(STORE), online=None)
    report = run_experiment(plan, gateway, lookup, tmp_path)
    assert report.acquired == 0 and report.replayed >= len(report.observations)
    assert (tmp_path / "report.json").exists()
    assert len(report.observations) == len(plan.prompt_ids) * len(plan.cases)
    assert all(o.parse_ok for o in report.observations), [
        o.parse_error for o in report.observations
    ]
    valid = [o for o in report.observations if o.case_id == "fixture-ct-g3-secuencias"]
    malformed = [o for o in report.observations if o.case_id == "fixture-malformed-grade-9"]
    assert all(o.status == "refused" and o.refusal_correct for o in malformed)
    assert all(o.metrics is not None and o.metrics.micro_skill_count == 0 for o in malformed)
    for observation in valid:
        assert observation.status == "ok"
        assert observation.metrics is not None
        assert observation.metrics.micro_skill_count >= 1
        assert observation.metrics.quote_support is not None
        # Foreign excerpts carry no numbered indicators, so coverage is absent, not zero.
        assert observation.metrics.indicator_coverage is None
        assert observation.judge_scores and all(0 <= s <= 1 for s in observation.judge_scores)
    frame = prompt_report.observations_frame(report)
    assert set(prompt_report.eligibility(frame)["prompt_id"]) == set(plan.prompt_ids)
    assert prompt_report.ranking(frame).height == len(plan.prompt_ids)
    assert prompt_report.worst_cells(frame).height == len(valid)
    second = run_experiment(plan, LLMGateway(ReplayStore(STORE)), lookup, tmp_path / "again")
    assert second.observations == report.observations
    assert (tmp_path / "report.json").read_bytes() == (
        tmp_path / "again" / "report.json"
    ).read_bytes()


def test_unrecorded_request_fails_instead_of_calling_a_model(tmp_path: Path):
    plan = load_plan(PLAN)
    lookup, _ = golden_lookup(GOLDEN)
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
    lookup, _ = golden_lookup(GOLDEN)
    context = render_context(plan.cases[0], lookup, plan.document_slugs)
    assert "precise and unambiguous instructions" in context and context.startswith("[")
    assert byte_digest(context.encode()) == byte_digest(
        render_context(plan.cases[0], lookup, plan.document_slugs).encode()
    )
    store = ReplayStore(STORE)
    records = [store.path(k).read_text() for k in store.keys()]
    # Records may carry short evidence quotes (model output) but never the rendered context.
    assert records and all(context not in r and '"user_content"' not in r for r in records)
    assert all("Authorization" not in r and "Bearer " not in r for r in records)


def test_verification_text_names_every_unsupported_quote():
    plan = load_plan(PLAN)
    lookup, _ = golden_lookup(GOLDEN)
    context = ContextIndex(render_context(plan.cases[0], lookup, plan.document_slugs))
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
    assert "1 de 2" in text and "NO SOSTENIDA en 'predecir-programa'" in text
    assert "cita inventada" in text and "no reportés" not in text


def test_load_outputs_returns_only_parsed_decompositions(tmp_path: Path):
    plan = load_plan(PLAN)
    lookup, _ = golden_lookup(GOLDEN)
    report = run_experiment(plan, LLMGateway(ReplayStore(STORE)), lookup, tmp_path)
    outputs = load_outputs(report, STORE)
    assert {case_id for _, _, case_id in outputs} == {"fixture-ct-g3-secuencias"}
    assert all(o.status == "ok" and o.micro_skills for o in outputs.values())
