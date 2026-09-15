from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from goes_tech_kg.eval import prompt_metrics
from goes_tech_kg.schemas.decomposition import DecompositionOutput
from goes_tech_kg.schemas.scope import TechnologyScope, load_scope

SCOPE = Path("config/technology_scope.yaml")


def test_scope_file_validates_and_ceilings_match():
    scope = load_scope(SCOPE)
    assert scope.unit(2, 5).technology_indicators == {
        "5.1",
        "5.2",
        "5.3",
        "5.4",
        "5.5",
        "5.6",
        "5.8",
        "5.9",
    }
    assert scope.unit(3, 1).ceiling == pytest.approx(0.333)
    assert scope.unit(6, 3).indicators["3.9"].indicator_class == "science_dependency"
    body = yaml.safe_load(SCOPE.read_text())
    body["coverage_ceilings"]["2/5"] = 1.0
    with pytest.raises(ValidationError, match="derived"):
        TechnologyScope.model_validate(body)
    body = yaml.safe_load(SCOPE.read_text())
    body["units"][0]["indicators"]["5.1"]["class"] = "maybe"
    with pytest.raises(ValidationError):
        TechnologyScope.model_validate(body)


def test_scoped_coverage_ignores_out_of_scope_indicators():
    context = (
        "[sv-cyt p.46 ¶a] 1.1. Obtiene valores para una misma magnitud.\n"
        "[sv-cyt p.46 ¶b] 1.2. Selecciona el instrumento de medición apropiado.\n"
        "[sv-cyt p.46 ¶c] 1.3. Efectúa un experimento que involucre fuerzas de contacto.\n"
        "[sv-cyt p.46 ¶d] 1.6. Comunica sus observaciones."
    )

    def micro(slug: str, quote: str) -> dict[str, object]:
        return {
            "slug": slug,
            "statement": "Mide una magnitud con distintos instrumentos.",
            "observable_verb": "Mide",
            "knowledge_object": "instrumentos de medición",
            "strand": "technical_systems",
            "cognitive_domain": "applying",
            "min_tier": "T0",
            "half_life": "DURABLE",
            "teacher_prep_level": 1,
            "evidence_of_mastery": "Tabla con tres mediciones.",
            "estimated_minutes": 45,
            "evidence_quotes": [{"quote": quote, "locator_hint": "[sv-cyt p.46 ¶a]"}],
        }

    output = DecompositionOutput.model_validate(
        {
            "status": "ok",
            "coverage_notes": "",
            "micro_skills": [
                micro("medir", "1.1. Obtiene valores para una misma magnitud."),
                micro("seleccionar", "1.2. Selecciona el instrumento de medición apropiado."),
            ],
        }
    )
    scope = load_scope(SCOPE).unit(3, 1)
    assert prompt_metrics.indicator_coverage(output, context) == pytest.approx(0.5)
    assert prompt_metrics.indicator_coverage(
        output, context, in_scope=scope.technology_indicators
    ) == pytest.approx(1.0)
    assert prompt_metrics.indicator_coverage(output, context, in_scope=frozenset()) is None
