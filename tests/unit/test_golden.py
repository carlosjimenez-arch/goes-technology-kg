from pathlib import Path

import pytest
from pydantic import ValidationError

from goes_tech_kg.eval import golden_metrics
from goes_tech_kg.schemas.decomposition import DecompositionOutput
from goes_tech_kg.schemas.golden import GoldenRecord, load_golden

GOLDEN_DIR = Path("evaluation/golden")


def micro(slug: str, quote: str, locator: str = "[sv-cyt p.41 ¶paragraph-p1-x]", prereqs=()):
    return {
        "slug": slug,
        "statement": "Construye un objeto técnico.",
        "observable_verb": "Construye",
        "knowledge_object": "objeto técnico",
        "strand": "technical_systems",
        "cognitive_domain": "applying",
        "min_tier": "T0",
        "half_life": "DURABLE",
        "teacher_prep_level": 1,
        "evidence_of_mastery": "Objeto funcional.",
        "estimated_minutes": 45,
        "prerequisites": list(prereqs),
        "evidence_quotes": [{"quote": quote, "locator_hint": locator}],
    }


def test_golden_records_load_and_only_human_records_can_gate():
    records = load_golden(GOLDEN_DIR)
    assert {r.case_id for r in records} >= {
        "sv-g2-u5-objetos-tecnicos",
        "sv-g4-u1-maquinas-energia",
        "ct-g3-secuencias-depuracion",
    }
    assert all(r.annotation_method == "llm_expert_cross_vendor" for r in records)
    with pytest.raises(ValueError, match="human_unaided is required"):
        load_golden(GOLDEN_DIR, require_human=True)
    body = records[0].model_dump(mode="json")
    body["annotation_method"] = "human_unaided"
    with pytest.raises(ValidationError, match="never provisional"):
        GoldenRecord.model_validate(body)
    body["status"] = "reviewed"
    assert GoldenRecord.model_validate(body).annotation_method == "human_unaided"
    bad = records[0].model_dump(mode="json")
    bad["edges"].append(
        {"source": "nope", "target": bad["micro_skills"][0]["id"], "justification": "x"}
    )
    with pytest.raises(ValidationError, match="distinct golden"):
        GoldenRecord.model_validate(bad)


def test_metrics_on_known_alignment():
    golden = next(g for g in load_golden(GOLDEN_DIR) if g.case_id == "sv-g2-u5-objetos-tecnicos")
    output = DecompositionOutput.model_validate(
        {
            "status": "ok",
            "coverage_notes": "",
            "micro_skills": [
                micro("balanza", "5.1. Construye una balanza."),
                micro("registrar", "5.2. Registra valores de magnitud", prereqs=["balanza"]),
                micro("puente", "5.3. Construye un modelo de puente rígido."),
                micro("optimizar", "5.6. Explica sus ideas", prereqs=["puente"]),
                micro("invento", "Sin indicador citado."),
            ],
        }
    )
    result = golden_metrics.score(output, golden)
    assert result["decomposition_recall"] == pytest.approx(4 / 8)
    assert result["decomposition_precision"] == pytest.approx(4 / 5)
    # balanza -> registrar is golden; puente -> optimizar is not.
    assert result["edge_precision"] == pytest.approx(0.5)
    assert result["edge_recall"] == pytest.approx(1 / 3)
    assert result["edge_f1"] == pytest.approx(0.4)
    assert result["cognitive_agreement"] == pytest.approx(3 / 4)
    assert "g2-usar-terrario" in result["unmatched_golden"]
    empty_edges = DecompositionOutput.model_validate(
        {"status": "ok", "coverage_notes": "", "micro_skills": [micro("balanza", "5.1. Construye")]}
    )
    zero = golden_metrics.score(empty_edges, golden)
    assert zero["edge_precision"] is None and zero["edge_f1"] == 0.0
    b0 = golden_metrics.official_baseline(golden)
    assert b0["decomposition_recall"] == 1.0 and b0["edge_recall"] == 0.0
    ct = next(g for g in load_golden(GOLDEN_DIR) if g.case_id == "ct-g3-secuencias-depuracion")
    assert golden_metrics.official_baseline(ct)["decomposition_recall"] == 0.0


def test_paragraph_keys_align_foreign_evidence():
    ct = next(g for g in load_golden(GOLDEN_DIR) if g.case_id == "ct-g3-secuencias-depuracion")
    output = DecompositionOutput.model_validate(
        {
            "status": "ok",
            "coverage_notes": "",
            "micro_skills": [
                micro(
                    "predecir",
                    "use logical reasoning to predict the behaviour of simple programs",
                    "[eng-computing html ¶paragraph-p94-f79248dc0d11c7ae]",
                )
            ],
        }
    )
    assert golden_metrics.align(output, ct) == {"predecir": frozenset({"ct3-predecir-resultado"})}
    merged = DecompositionOutput.model_validate(
        {
            "status": "ok",
            "coverage_notes": "",
            "micro_skills": [
                {
                    **micro("localizar-y-corregir", "Identify where a bug has occurred"),
                    "evidence_quotes": [
                        {
                            "quote": "Identify where a bug has occurred",
                            "locator_hint": "[u html ¶paragraph-p167-a]",
                        },
                        {
                            "quote": "Repeat step 4 until the program is free of bugs",
                            "locator_hint": "[u html ¶paragraph-p377-b]",
                        },
                    ],
                }
            ],
        }
    )
    # One merged system skill covers both golden items it cites.
    assert golden_metrics.score(merged, ct)["decomposition_recall"] == pytest.approx(2 / 8)
