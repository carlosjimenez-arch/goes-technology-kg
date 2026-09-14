import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from goes_tech_kg.curriculum.dependencies import cross_subject_conflicts
from goes_tech_kg.graph.diff import diff_graphs
from goes_tech_kg.schemas.base import canonical_json, stable_id
from goes_tech_kg.schemas.curriculum import BudgetLedger, BudgetPolicy, ContextTag
from goes_tech_kg.schemas.graph import GraphSnapshot
from goes_tech_kg.schemas.skills import Edge, MicroSkill, SkillPlacement, ToolBinding


def graph(skill, micros, edges=()):
    return GraphSnapshot(
        version="v1",
        skills=(skill,),
        micro_skills=tuple(micros),
        edges=tuple(edges),
        source_ids=("source-fixture",),
    )


def test_context_is_not_a_graph_field(micro_factory):
    with pytest.raises(ValidationError, match="Extra inputs"):
        micro_factory(contexts=[{"type": "everyday", "domain": "economic"}])
    assert ContextTag(type="everyday", domain="economic").schema_version == "context-tag/1.0"


def test_volatile_cannot_enter_graph_or_be_a_prerequisite(skill, micro_factory, evidence):
    volatile = micro_factory(half_life="VOLATILE")
    target = micro_factory(
        slug="debug",
        observable_verb="Debug",
        prerequisites=[{"id": volatile.id, "type": "PREREQUISITE"}],
    )
    edge = Edge(
        source=volatile.id,
        target=target.id,
        type="PREREQUISITE",
        justification="Fixture dependency",
        source_refs=(evidence,),
    )
    with pytest.raises(ValidationError, match="VOLATILE"):
        graph(skill, [volatile, target], [edge])
    with pytest.raises(ValidationError, match="Extra inputs"):
        ToolBinding(
            id="binding-editor",
            micro_skill_id=target.id,
            product="Editor",
            interface_version="1",
            instructions="Run program",
            min_tier="T1",
            prerequisites=[target.id],
        )


def test_material_and_preparation_requirements(micro_factory):
    for changes in [
        {"min_tier": "T4"},
        {"teacher_prep_level": 4},
        {"estimated_minutes": True},
        {"confidence": float("nan")},
        {"ct_dimension": None},
    ]:
        with pytest.raises(ValidationError):
            micro_factory(**changes)
    data = micro_factory().model_dump()
    del data["min_tier"]
    with pytest.raises(ValidationError):
        MicroSkill.model_validate(data)


def test_downgrade_must_be_one_tier_and_honest(micro_factory):
    variant = dict(
        target_tier="T0",
        method="Paper trace",
        evidence_of_mastery="Explain trace",
        estimated_minutes=45,
        preserved_constructs=["execution"],
        lost_performances=["editor operation"],
        equivalence="same_construct_candidate",
    )
    with pytest.raises(ValidationError, match="lost performances"):
        micro_factory(min_tier="T1", downgrade_variant=variant)
    variant.update(
        equivalence="conceptual_proxy_only", practical_bridge="Supervised editor practice"
    )
    assert (
        micro_factory(min_tier="T1", downgrade_variant=variant).downgrade_variant.estimated_minutes
        == 45
    )
    with pytest.raises(ValidationError, match="immediately lower"):
        micro_factory(min_tier="T2", downgrade_variant=variant)


def test_semantic_diff_and_identity(skill, micro_factory):
    before = micro_factory()
    after = micro_factory(estimated_minutes=45)
    assert before.id == after.id
    result = diff_graphs(graph(skill, [before]), graph(skill, [after]))
    assert result.nodes.modified[0].changed_fields == ("estimated_minutes",)
    changed = micro_factory(knowledge_object="Parallel execution")
    assert changed.id != before.id
    result = diff_graphs(graph(skill, [before]), graph(skill, [changed]))
    assert len(result.nodes.added) == len(result.nodes.removed) == 1
    with pytest.raises(ValidationError, match="ID"):
        micro_factory(id="micro-fabricated")


def test_prerequisites_mirror_edges_and_cycles_fail(skill, micro_factory, evidence):
    a = micro_factory()
    b = micro_factory(slug="debug", observable_verb="Debug")
    with pytest.raises(ValidationError, match="mirror"):
        graph(
            skill,
            [
                a,
                micro_factory(
                    slug="debug",
                    observable_verb="Debug",
                    prerequisites=[{"id": a.id, "type": "PREREQUISITE"}],
                ),
            ],
        )
    a = micro_factory(prerequisites=[{"id": b.id, "type": "PREREQUISITE"}])
    b = micro_factory(
        slug="debug", observable_verb="Debug", prerequisites=[{"id": a.id, "type": "PREREQUISITE"}]
    )
    edges = [
        Edge(
            source=x.id,
            target=y.id,
            type="PREREQUISITE",
            justification="Cycle fixture",
            source_refs=(evidence,),
        )
        for x, y in [(a, b), (b, a)]
    ]
    with pytest.raises(ValidationError, match="cycle"):
        graph(skill, [a, b], edges)


def test_corequisites_are_symmetric_not_strict_cycles(skill, micro_factory, evidence):
    a = micro_factory()
    b = micro_factory(slug="debug", observable_verb="Debug")
    a = micro_factory(prerequisites=[{"id": b.id, "type": "CO_REQUISITE"}])
    b = micro_factory(
        slug="debug", observable_verb="Debug", prerequisites=[{"id": a.id, "type": "CO_REQUISITE"}]
    )
    edge = Edge(
        source=a.id,
        target=b.id,
        type="CO_REQUISITE",
        justification="Learn together",
        source_refs=(evidence,),
    )
    assert graph(skill, [a, b], [edge]).edges[0].source == min(a.id, b.id)


def test_external_grade_conflict(skill, micro_factory, evidence):
    dependency = {
        "repository": "goes-math-kg",
        "snapshot_sha256": "b" * 64,
        "node_id": "coordinates",
        "available_grade": 5,
        "source_refs": [evidence.model_dump()],
    }
    m = micro_factory(cross_subject_deps=[dependency])
    dep = m.cross_subject_deps[0]
    edge = Edge(
        source=dep.qualified_id,
        target=m.id,
        type="CROSS_SUBJECT_PREREQUISITE",
        justification="Needs coordinates",
        source_refs=(evidence,),
    )
    snapshot = graph(skill, [m], [edge])
    conflict = cross_subject_conflicts(snapshot, (SkillPlacement(micro_skill_id=m.id, grade=3),))
    assert conflict[0]["available_grade"] == 5 and conflict[0]["requested_grade"] == 3
    assert cross_subject_conflicts(snapshot, (SkillPlacement(micro_skill_id=m.id, grade=5),)) == []


@pytest.mark.parametrize("mode", ["standalone", "transversal", "hybrid"])
def test_budget_mode_capacity(mode):
    own = 30 if mode != "transversal" else 0
    host = {"mathematics": 20} if mode != "standalone" else {}
    policy = BudgetPolicy(
        mode=mode,
        grade=3,
        own_minutes=own,
        borrowed_minutes_cap=sum(host.values()),
        host_capacities=host,
    )
    charges = []
    if own:
        charges.append(dict(micro_skill_id="trace", minutes=own))
    if host:
        charges.append(
            dict(
                micro_skill_id="data",
                minutes=20,
                host_subject="mathematics",
                host_indicator="table-reading",
            )
        )
    assert len(BudgetLedger(policy=policy, charges=charges).charges) == len(charges)
    charges[-1]["minutes"] += 1
    with pytest.raises(ValidationError, match="capacity"):
        BudgetLedger(policy=policy, charges=charges)


@given(st.dictionaries(st.text(alphabet="abc", min_size=1, max_size=4), st.integers(), max_size=10))
def test_identity_is_dictionary_order_independent(payload):
    reordered = dict(reversed(list(payload.items())))
    assert stable_id("skill", "test", payload) == stable_id("skill", "test", reordered)
    assert canonical_json(payload) == canonical_json(reordered)


def test_curriculum_unit_identity_and_totals():
    from goes_tech_kg.schemas.curriculum import CurriculumUnit

    data = dict(
        slug="sequence-unit",
        grade=2,
        label="Sequences",
        micro_skill_ids=("b", "a"),
        instruction_minutes=20,
        assessment_minutes=5,
        setup_minutes=5,
        review_minutes=0,
        total_minutes=30,
    )
    unit = CurriculumUnit(**data)
    assert unit.id.startswith("unit-sequence-unit-")
    assert unit.micro_skill_ids == ("a", "b")
    assert CurriculumUnit(**{**data, "micro_skill_ids": ("a", "b")}).id == unit.id
    with pytest.raises(ValidationError, match="inconsistent"):
        CurriculumUnit(**{**data, "total_minutes": 20})
    with pytest.raises(ValidationError, match="identity"):
        CurriculumUnit(**{**data, "id": "forged"})


@given(st.permutations((0, 1, 2, 3)))
def test_graph_serialization_is_invariant_to_node_order(order):
    from goes_tech_kg.schemas.skills import Skill

    nodes = tuple(Skill(slug=f"skill-{i}", label=f"Skill {i}") for i in range(4))
    left = GraphSnapshot(version="v1", skills=nodes)
    right = GraphSnapshot(version="v1", skills=tuple(nodes[i] for i in order))
    assert canonical_json(left) == canonical_json(right)
    assert not diff_graphs(left, right).nodes.added
