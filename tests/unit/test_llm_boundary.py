import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from goes_tech_kg.eval import prompt_metrics
from goes_tech_kg.eval.prompt_experiment import ExperimentCase, ModelSpec, build_request
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.llm.replay import ReplayMiss, ReplayStore
from goes_tech_kg.llm.vertex import endpoint, request_body
from goes_tech_kg.prompts.decomposition import DECOMPOSITION_GUIDED, DECOMPOSITION_PROMPTS
from goes_tech_kg.prompts.judge import CURRICULAR_JUDGE
from goes_tech_kg.prompts.registry import Prompt
from goes_tech_kg.schemas.base import byte_digest
from goes_tech_kg.schemas.decomposition import DecompositionOutput, JudgeOutput
from goes_tech_kg.schemas.llm import GenerationSettings, LLMRequest, ResponseRecord, TokenUsage

CONTEXT = (
    "[demo p.1 ¶p1] 5.1 . Construye una balanza.\n"
    "[demo p.1 ¶p2] 5. 2. Registra valores de magnitud empleando un instrumento de medición."
)


def case(**updates):
    data = dict(id="demo-case", grade="2", skill_map_entry="Construir una balanza y registrar.")
    data.update(updates)
    return ExperimentCase.model_validate(data)


def spec(**updates):
    data = dict(model="gemini-2.5-flash", location="us-central1")
    data.update(updates)
    return ModelSpec.model_validate(data)


def micro(**updates):
    data = {
        "slug": "construir-balanza",
        "statement": "Construye una balanza de brazos iguales con materiales del aula.",
        "observable_verb": "Construye",
        "knowledge_object": "balanza de brazos iguales",
        "strand": "technical_systems",
        "cognitive_domain": "applying",
        "min_tier": "T0",
        "half_life": "DURABLE",
        "teacher_prep_level": 1,
        "evidence_of_mastery": "La balanza equilibra dos masas iguales.",
        "estimated_minutes": 45,
        "evidence_quotes": [
            {"quote": "5.1 . Construye una balanza.", "locator_hint": "[demo p.1 ¶p1]"}
        ],
    }
    data.update(updates)
    return data


def test_prompt_version_derives_from_text_and_rendering_is_strict():
    base = DECOMPOSITION_GUIDED
    changed = Prompt(
        id=base.id,
        system=base.system + " ",
        user_template=base.user_template,
        output_schema_version=base.output_schema_version,
        technique=base.technique,
    )
    assert base.version != changed.version and base.version.startswith("decomposition-guided@")
    assert len({p.version for p in DECOMPOSITION_PROMPTS.values()}) == 6
    with pytest.raises(ValueError, match="variables mismatch"):
        base.render(case_id="x")
    rendered = base.render(
        case_id="x", grade="2", skill_map_entry="s", context="c", output_schema_version="v"
    )
    assert "{{" not in rendered and "decomposition/1.0" not in rendered
    from goes_tech_kg.prompts.judge import CURRICULAR_JUDGE_V2

    assert CURRICULAR_JUDGE_V2.system != CURRICULAR_JUDGE.system
    assert "verification" in CURRICULAR_JUDGE_V2.variables
    for prompt in (*DECOMPOSITION_PROMPTS.values(), CURRICULAR_JUDGE, CURRICULAR_JUDGE_V2):
        assert "Respondé" in prompt.system or "Respondé" in prompt.user_template


def test_request_key_covers_every_effective_field_and_hides_context():
    request, user_content = build_request(DECOMPOSITION_GUIDED, spec(), case(), CONTEXT, 0)
    assert request.user_content_sha256 == byte_digest(user_content.encode())
    assert "Construye una balanza" not in json.dumps(request.model_dump(mode="json"))
    baseline = request.key
    variants = [
        build_request(DECOMPOSITION_GUIDED, spec(model="gemini-2.5-pro"), case(), CONTEXT, 0),
        build_request(DECOMPOSITION_GUIDED, spec(location="global"), case(), CONTEXT, 0),
        build_request(DECOMPOSITION_GUIDED, spec(), case(), CONTEXT + " ", 0),
        build_request(DECOMPOSITION_GUIDED, spec(), case(), CONTEXT, 1),
        build_request(
            DECOMPOSITION_GUIDED,
            spec(settings=GenerationSettings(thinking_budget=1024)),
            case(),
            CONTEXT,
            0,
        ),
        build_request(DECOMPOSITION_PROMPTS["decomposition-fewshot"], spec(), case(), CONTEXT, 0),
    ]
    keys = {v[0].key for v in variants}
    assert baseline not in keys and len(keys) == len(variants)
    again, _ = build_request(DECOMPOSITION_GUIDED, spec(), case(), CONTEXT, 0)
    assert again.key == baseline
    with pytest.raises(ValidationError, match="not both"):
        GenerationSettings(thinking_budget=1, thinking_level="low")


def test_vertex_body_is_exactly_the_recorded_request():
    request, user_content = build_request(
        DECOMPOSITION_GUIDED,
        spec(settings=GenerationSettings(thinking_budget=4096)),
        case(),
        CONTEXT,
        0,
    )
    body = request_body(request, user_content)
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 4096}
    assert body["generationConfig"]["responseJsonSchema"] == DecompositionOutput.model_json_schema()
    assert body["contents"][0]["parts"][0]["text"] == user_content
    assert endpoint("p", "global", "m").startswith(
        "https://aiplatform.googleapis.com/v1/projects/p/"
    )
    assert endpoint("p", "us-central1", "m").startswith(
        "https://us-central1-aiplatform.googleapis.com/"
    )
    level, _ = build_request(
        DECOMPOSITION_GUIDED,
        spec(settings=GenerationSettings(thinking_level="medium")),
        case(),
        CONTEXT,
        0,
    )
    assert request_body(level, user_content)["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "medium"
    }


def record_for(request: LLMRequest, text: str) -> ResponseRecord:
    return ResponseRecord(
        request_sha256=request.key,
        request=request,
        response_text=text,
        response_sha256=byte_digest(text.encode()),
        provider_model_version="gemini-2.5-flash",
        finish_reason="STOP",
        usage=TokenUsage(prompt_tokens=10, output_tokens=5),
        latency_ms=1,
    )


def test_replay_store_is_immutable_and_replay_never_calls_a_model(tmp_path: Path):
    store = ReplayStore(tmp_path)
    request, user_content = build_request(DECOMPOSITION_GUIDED, spec(), case(), CONTEXT, 0)
    gateway = LLMGateway(store, online=None)
    with pytest.raises(ReplayMiss):
        gateway.complete(request, user_content)
    record = record_for(request, '{"status":"refused"}')
    store.put(record)
    assert gateway.complete(request, user_content) == record and gateway.replayed == 1
    with pytest.raises(ValueError, match="refusing to overwrite"):
        store.put(record_for(request, '{"status":"ok"}'))
    tampered = json.loads(store.path(request.key).read_text())
    tampered["response_text"] = "changed"
    store.path(request.key).write_text(json.dumps(tampered))
    with pytest.raises(ValidationError, match="digest"):
        store.get(request)
    with pytest.raises(ValidationError, match="request digest"):
        ResponseRecord.model_validate(
            {**record.model_dump(mode="json"), "request_sha256": "0" * 64}
        )


def test_decomposition_contract_and_metrics():
    ok = DecompositionOutput.model_validate(
        {
            "status": "ok",
            "coverage_notes": "Indicadores 5.1 y 5.2 cubiertos.",
            "micro_skills": [
                micro(),
                micro(
                    slug="registrar-magnitud",
                    statement="Registra valores de magnitud con un instrumento propio.",
                    observable_verb="Registra",
                    knowledge_object="valores de magnitud",
                    prerequisites=["construir-balanza"],
                    evidence_quotes=[
                        {
                            "quote": "Registra valores de magnitud empleando un instrumento",
                            "locator_hint": "[demo p.1 ¶p2]",
                        },
                        {
                            "quote": "cita que no existe en el contexto",
                            "locator_hint": "[demo p.9 ¶zz]",
                        },
                    ],
                ),
            ],
        }
    )
    metrics = prompt_metrics.summarize(ok, CONTEXT)
    assert metrics["micro_skill_count"] == 2
    assert metrics["quote_exactness"] == pytest.approx(2 / 3)
    assert metrics["locator_exactness"] == pytest.approx(2 / 3)
    assert metrics["indicator_coverage"] == 0.5 and metrics["t0_share"] == 1.0
    assert metrics["observable_rate"] == 1.0 and metrics["duplicate_rate"] == 0.0
    assert metrics["mean_cognitive_level"] == 1.0 and metrics["prerequisite_count"] == 1
    refused = DecompositionOutput.model_validate(
        {
            "status": "refused",
            "refusal_reason": "grado fuera de rango",
            "micro_skills": [],
            "coverage_notes": "",
        }
    )
    with pytest.raises(ValidationError, match="at least one micro-skill"):
        DecompositionOutput.model_validate({"status": "ok", "coverage_notes": ""})
    assert prompt_metrics.summarize(refused, CONTEXT)["quote_exactness"] is None
    assert prompt_metrics.jaccard(ok, ok) == 1.0 and prompt_metrics.jaccard(ok, refused) == 0.0
    for bad in [
        {"status": "refused", "micro_skills": [micro()], "coverage_notes": ""},
        {"status": "ok", "micro_skills": [], "coverage_notes": ""},
        {"status": "ok", "micro_skills": [micro(), micro()], "coverage_notes": ""},
        {
            "status": "ok",
            "micro_skills": [micro(prerequisites=["construir-balanza"])],
            "coverage_notes": "",
        },
        {"status": "ok", "micro_skills": [micro(min_tier="T2")], "coverage_notes": ""},
        {"status": "ok", "micro_skills": [micro(half_life="VOLATILE")], "coverage_notes": ""},
        {
            "status": "ok",
            "micro_skills": [micro(strand="computational_thinking")],
            "coverage_notes": "",
        },
        {"status": "ok", "micro_skills": [micro(slug="Con Espacios")], "coverage_notes": ""},
    ]:
        with pytest.raises(ValidationError):
            DecompositionOutput.model_validate(bad)
    weak = DecompositionOutput.model_validate(
        {"status": "ok", "micro_skills": [micro(observable_verb="Comprende")], "coverage_notes": ""}
    )
    assert prompt_metrics.observable_rate(weak) == 0.0
    with pytest.raises(ValidationError, match="accept requires"):
        JudgeOutput(verdict="accept", score=0.5, rationale="x")
    with pytest.raises(ValidationError, match="reject requires"):
        JudgeOutput(verdict="reject", score=0.7, rationale="x")
