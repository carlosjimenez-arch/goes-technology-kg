"""Real golden release workload: graph validation and full curriculum allocation."""

import json
from pathlib import Path

import pytest

from goes_tech_kg.curriculum.publish import compile_graph
from goes_tech_kg.curriculum.schedule import schedule
from goes_tech_kg.eval.release_fixture import release_fixture
from goes_tech_kg.schemas.release import ReleaseScenario, TeachingUnit

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release_workload(tmp_path):
    evidence, plan = release_fixture(ROOT, tmp_path / "data")
    row = json.loads(
        (ROOT / "data/processed/release_fixture/g2-release-golden-design.json").read_text()
    )
    return evidence, plan, row


def test_compile_evidenced_graph(benchmark, release_workload):
    evidence, _, row = release_workload
    graph, rows, _, quarantine = benchmark(compile_graph, [row], "v2", evidence)
    assert len(graph.micro_skills) == len(rows) == 3 and not quarantine


def test_schedule_complete_unit(benchmark, release_workload):
    evidence, plan, row = release_workload
    _, _, ids, _ = compile_graph([row], "v2", evidence)
    unit = TeachingUnit.model_validate(row["revised"])
    result = benchmark(
        schedule, (unit,), (plan,), ids, ReleaseScenario(minutes_per_grade=1000, mode="standalone")
    )
    assert result["status"] == "capacity_feasible"
    assert (
        sum(c["minutes"] for c in result["ledgers"][0]["rows"])
        == result["grades"][0]["required_minutes"]
    )
