"""Offline release reproduction from actual source excerpts and recorded Vertex responses."""

import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from goes_tech_kg.curriculum.external import verify_external_dependency
from goes_tech_kg.curriculum.generate import checks, generate_units, serving_schema, versioned_body
from goes_tech_kg.curriculum.publish import build_release, compile_graph
from goes_tech_kg.curriculum.schedule import schedule, topological_order
from goes_tech_kg.eval.release_fixture import release_fixture
from goes_tech_kg.llm.gateway import LLMGateway
from goes_tech_kg.llm.replay import ReplayStore
from goes_tech_kg.retrieval.release_context import EvidenceIndex
from goes_tech_kg.schemas.base import byte_digest, canonical_json
from goes_tech_kg.schemas.release import ReleaseScenario, TeachingUnit

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release_case(tmp_path):
    evidence, plan = release_fixture(ROOT, tmp_path / "data")
    row = json.loads(
        (ROOT / "data/processed/release_fixture/g2-release-golden-design.json").read_text()
    )
    return evidence, plan, row


def test_recorded_workflow_and_byte_identical_publication(tmp_path, release_case):
    evidence, plan, row = release_case
    gateway = LLMGateway(ReplayStore(ROOT / "data/processed/llm_responses"))
    output = tmp_path / "data/processed/curriculum_experiment"
    generate_units((plan,), evidence, gateway, output, workers=1)
    actual = json.loads((output / (plan.id + ".json")).read_text())
    assert actual == row
    assert gateway.acquired == 0 and gateway.replayed == 3
    generate_units((plan,), evidence, gateway, output, workers=1)
    assert gateway.replayed == 3  # Completed proposals resume from verified immutable records.
    (tmp_path / "config").mkdir()
    (tmp_path / "config/curriculum_map.json").write_text(
        canonical_json({"units": [plan.model_dump(mode="json")]})
    )
    (tmp_path / "config/release_scenarios.json").write_bytes(
        (ROOT / "config/release_scenarios.json").read_bytes()
    )
    (tmp_path / "uv.lock").write_bytes((ROOT / "uv.lock").read_bytes())
    replay_dir = tmp_path / "data/processed/llm_responses"
    replay_dir.mkdir()
    for key in row["request_keys"]:
        (replay_dir / (key + ".json")).write_bytes(
            (ROOT / "data/processed/llm_responses" / (key + ".json")).read_bytes()
        )
    first = build_release(tmp_path, evidence)
    release = tmp_path / "data/processed/curriculum_releases"
    before = {p.relative_to(release): p.read_bytes() for p in release.rglob("*") if p.is_file()}
    assert build_release(tmp_path, evidence) == first
    assert before == {
        p.relative_to(release): p.read_bytes() for p in release.rglob("*") if p.is_file()
    }
    assert "Design" in (release / "v2/curriculum_proposal.html").read_text()
    evaluation = json.loads((release / "v2/evaluation.json").read_text())
    assert evaluation["micro_skills"] == 3 and evaluation["quarantined"] == 0
    assert first["adoption_ready"] is False
    for name, sha in first["artifacts_sha256"].items():
        assert byte_digest((release / name).read_bytes()) == sha


@pytest.mark.parametrize("mode,share", [("standalone", 1), ("transversal", 0), ("hybrid", 0.1)])
def test_full_budget_accounting_and_host_traceability(release_case, mode, share):
    evidence, plan, row = release_case
    _, _, ids, _ = compile_graph([row], "v2", evidence)
    unit = TeachingUnit.model_validate(row["revised"])
    scenario = ReleaseScenario(
        minutes_per_grade=1000,
        mode=mode,
        own_share=share,
        host_capacities={} if mode == "standalone" else {"science": 900},
        host_indicator=None if mode == "standalone" else "proposed-design-integration",
    )
    result = schedule((unit,), (plan,), ids, scenario)
    assert result["status"] == "capacity_feasible"
    required = (
        sum(m.estimated_minutes for m in unit.micro_skills)
        + unit.assessment_minutes
        + unit.review_minutes
        + unit.setup_minutes
    )
    assert result["grades"][0]["required_minutes"] == required
    charges = result["ledgers"][0]["rows"]
    assert sum(c["minutes"] for c in charges) == required
    assert [c["start_minute"] for c in charges] == [
        sum(previous["minutes"] for previous in charges[:i]) for i in range(len(charges))
    ]
    if share < 1:
        borrowed = [c for c in charges if c["host_subject"]]
        assert borrowed and all(c["host_indicator"] for c in borrowed)
    assert result["deployable"] is False


def test_time_resource_and_coverage_failures(release_case):
    evidence, plan, row = release_case
    _, _, ids, _ = compile_graph([row], "v2", evidence)
    unit = TeachingUnit.model_validate(row["revised"])
    result = schedule(
        (unit,),
        (plan,),
        ids,
        ReleaseScenario(minutes_per_grade=1, mode="standalone", available_resources=()),
    )
    assert result["status"] == "infeasible"
    assert {r["code"] for r in result["diagnostics"]} >= {
        "missing_resources",
        "insufficient_time_or_host_capacity",
    }
    with pytest.raises(ValueError, match="coverage"):
        schedule((), (plan,), ids, ReleaseScenario(minutes_per_grade=100, mode="standalone"))


def test_evidence_tampering_and_unresolved_prerequisite_are_rejected(tmp_path, release_case):
    evidence, plan, row = release_case
    unit = TeachingUnit.model_validate(row["revised"])
    refs = evidence.select(plan)
    assert not checks(unit, plan, refs, evidence)
    broken = unit.model_dump()
    broken["micro_skills"][0]["prerequisites"] = ["missing-node"]
    assert any(
        "unresolved prerequisite" in e
        for e in checks(TeachingUnit.model_validate(broken), plan, refs, evidence)
    )
    row["revised"] = broken
    graph, _, _, quarantine = compile_graph([row], "v2", evidence)
    assert len(graph.micro_skills) < 3 and quarantine
    with pytest.raises(ValueError, match="exactly one"):
        evidence.resolve_quote("a fabricated quotation", "unknown", refs)
    with pytest.raises(ValueError, match="unsupported quotation"):
        evidence.resolve_quote("a fabricated quotation", refs[0].paragraph_ids[0], refs)
    path = tmp_path / "data/interim/chunks.jsonl"
    chunks = [json.loads(line) for line in path.read_text().splitlines()]
    chunks[0]["text"] = chunks[0]["text"].replace("design", "xxxxx")
    path.write_text("".join(canonical_json(c) + "\n" for c in chunks))
    with pytest.raises(ValueError):
        EvidenceIndex(tmp_path / "data")


def test_format_repairs_are_logged_and_do_not_create_prerequisites():
    repairs = []
    body = versioned_body(
        {"schema_version": "1.0", "human_validation_required": False},
        "unit-review/1.0",
        "test",
        repairs,
    )
    assert body["human_validation_required"] is True and len(repairs) == 2
    repairs = []
    body = versioned_body(
        {"micro_skills": [{"slug": "Trace_Path", "prerequisites": ["unknown"]}]},
        "teaching-unit/1.0",
        "test",
        repairs,
    )
    assert body["micro_skills"][0] == {"slug": "trace-path", "prerequisites": ["unknown"]}
    assert repairs
    assert serving_schema({"properties": {"title": {"type": "string", "minLength": 1}}}) == {
        "properties": {"title": {"type": "string"}}
    }
    with pytest.raises(ValueError, match="unsupported generated"):
        versioned_body({"schema_version": "future"}, "unit-review/1.0", "test", [])


@given(st.permutations(["a", "b", "c", "d"]))
def test_topological_order_is_permutation_independent(keys):
    dependencies = {"a": set(), "b": {"a"}, "c": {"a"}, "d": {"b", "c"}}
    assert topological_order({k: dependencies[k] for k in keys}) == ("a", "b", "c", "d")


@pytest.mark.parametrize("dependencies", [{"a": {"b"}}, {"a": {"b"}, "b": {"a"}}])
def test_bad_prerequisite_projection(dependencies):
    with pytest.raises(ValueError):
        topological_order(dependencies)


def test_real_external_snapshot_detects_grade_conflict():
    path = ROOT / "data/processed/external_catalogs/science-provisional.json"
    data = json.loads(path.read_text())
    node = next(n for n in data["nodes"] if n["available_grade"] == 5)
    result = verify_external_dependency(path, byte_digest(path.read_bytes()), node["id"], 3)
    assert result["conflict"] is True and result["available_grade"] == 5
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_external_dependency(path, "0" * 64, node["id"], 3)
    with pytest.raises(ValueError, match="absent"):
        verify_external_dependency(path, byte_digest(path.read_bytes()), "absent", 3)


def test_modified_proposal_cannot_impersonate_model_record(release_case):
    from goes_tech_kg.curriculum.provenance import verify_record

    _, _, row = release_case
    verify_record(row, ROOT / "data/processed/llm_responses")
    row["revised"]["rationale"] = "Undocumented editorial replacement"
    with pytest.raises(ValueError, match="differs from recorded"):
        verify_record(row, ROOT / "data/processed/llm_responses")


def test_secondary_source_cannot_justify_primary_placement(release_case):
    evidence, plan, _ = release_case
    refs = evidence.select(plan)
    ref = next(r for r in refs if evidence.slugs[r.document_id] == "eng-computing")
    pid = ref.paragraph_ids[0]
    _, paragraph = evidence.paragraphs[pid]
    paragraph["original"]["section"] = "Key stage 3"
    with pytest.raises(ValueError, match="secondary-stage"):
        evidence.resolve_quote(" ".join(paragraph["text"].split()[:5]), pid, refs)
    assert all(pid not in r.paragraph_ids for r in evidence.select(plan))


def test_citation_offsets_resolve_original_crlf_text(release_case):
    from goes_tech_kg.corpus.normalize import normalize
    from goes_tech_kg.schemas.corpus import Paragraph

    evidence, plan, _ = release_case
    refs = evidence.select(plan)
    pid = refs[0].paragraph_ids[0]
    chunk, existing = evidence.paragraphs[pid]
    original = Paragraph.model_validate(
        {**existing["original"], "text": "Heading\r\nExact cited phrase."}
    )
    evidence.paragraphs[pid] = (chunk, normalize(original).model_dump(mode="json"))
    ref = evidence.resolve_quote("Exact cited phrase.", pid, refs)
    assert ref.start == 9 and original.text[ref.start : ref.end] == "Exact cited phrase."


def test_critique_of_initial_draft_cannot_impersonate_final_audit(release_case):
    from goes_tech_kg.curriculum.provenance import verify_record

    evidence, _, row = release_case
    row["final_review"] = row["review"]
    with pytest.raises(ValueError, match="does not evaluate the current revision"):
        verify_record(row, ROOT / "data/processed/llm_responses", evidence)


def test_explicit_curation_removes_only_reviewed_hypotheses():
    from goes_tech_kg.curriculum.curate import curate
    from goes_tech_kg.schemas.release import CurriculumCuration

    row = json.loads(
        (ROOT / "data/processed/curriculum_experiment/g6-technical-systems.json").read_text()
    )
    unit = TeachingUnit.model_validate(row["revised"])
    rules = CurriculumCuration.model_validate_json(
        (ROOT / "config/curriculum_curation.json").read_bytes()
    )
    corrected = curate(unit, rules)
    assert sum(len(m.prerequisites) for m in unit.micro_skills) == 2
    assert sum(len(m.prerequisites) for m in corrected.micro_skills) == 0
    assert [m.evidence_quotes for m in corrected.micro_skills] == [
        m.evidence_quotes for m in unit.micro_skills
    ]
    with pytest.raises(ValueError, match="stale"):
        curate(corrected, rules)
