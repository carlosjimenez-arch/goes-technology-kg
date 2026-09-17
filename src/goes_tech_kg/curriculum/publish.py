"""Build graph snapshots, pedagogical tables, critiques and immutable release manifests."""

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from goes_tech_kg.corpus.provenance import build_provenance
from goes_tech_kg.curriculum.curate import curate
from goes_tech_kg.curriculum.provenance import verify_record
from goes_tech_kg.curriculum.render import render_proposal
from goes_tech_kg.curriculum.schedule import schedule, topological_order
from goes_tech_kg.graph.diff import diff_graphs
from goes_tech_kg.retrieval.release_context import EvidenceIndex
from goes_tech_kg.schemas.base import byte_digest, canonical_json
from goes_tech_kg.schemas.graph import GraphSnapshot
from goes_tech_kg.schemas.llm import ContextRef
from goes_tech_kg.schemas.release import CurriculumCuration, ReleaseScenario, TeachingUnit, UnitPlan
from goes_tech_kg.schemas.skills import Edge, MicroSkill, PrerequisiteRef, Skill


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n")


def compile_graph(
    records: list[dict[str, Any]],
    version: str,
    evidence: EvidenceIndex,
    curation: CurriculumCuration | None = None,
) -> tuple[GraphSnapshot, list[dict[str, Any]], dict[tuple[str, str], str], list[dict[str, Any]]]:
    curation = curation or CurriculumCuration()
    skills: dict[str, Skill] = {}
    micros: dict[str, MicroSkill] = {}
    edges: dict[str, Edge] = {}
    rows = []
    quarantined = []
    identities: dict[tuple[str, str], str] = {}
    prepared = []
    for record in sorted(records, key=lambda r: (r["plan"]["grade"], r["plan"]["order"])):
        plan = UnitPlan.model_validate(record["plan"])
        key = "initial" if version == "v1" else record.get("selected_version", "revised")
        unit = TeachingUnit.model_validate(record[key])
        if version != "v1":
            unit = curate(unit, curation)
        if (
            unit.unit_id != plan.id
            or unit.grade != plan.grade
            or any(m.strand != plan.strand for m in unit.micro_skills)
        ):
            quarantined.append(
                {
                    "unit_id": unit.unit_id,
                    "reason": "unit identity, grade or strand differs from declared map",
                    "version": version,
                }
            )
            continue
        internal = {
            m.slug: set(m.prerequisites) & {s.slug for s in unit.micro_skills}
            for m in unit.micro_skills
        }
        try:
            topological_order(internal)
        except ValueError:
            quarantined.append(
                {
                    "unit_id": unit.unit_id,
                    "reason": "cyclic prerequisite proposal",
                    "version": version,
                }
            )
            continue
        parent = Skill(slug=plan.id, label=plan.title)
        skills[parent.id] = parent
        raw_refs = record["context_refs"]
        if version == "v1":
            raw_refs = record.get(
                "initial_context_refs", next(iter(record.get("context_history", [])), raw_refs)
            )
        refs = tuple(ContextRef.model_validate(r) for r in raw_refs)
        for micro in unit.micro_skills:
            resolved = []
            for quote in micro.evidence_quotes:
                try:
                    resolved.append(evidence.resolve_quote(quote.quote, quote.locator_hint, refs))
                except ValueError as exc:
                    quarantined.append(
                        {
                            "unit_id": unit.unit_id,
                            "slug": micro.slug,
                            "reason": str(exc),
                            "version": version,
                        }
                    )
            if not resolved:
                continue
            node = MicroSkill(
                slug=micro.slug,
                parent_skill_id=parent.id,
                observable_verb=micro.observable_verb,
                knowledge_object=micro.knowledge_object,
                strand=micro.strand,
                cognitive_domain=micro.cognitive_domain,
                ct_dimension=micro.ct_dimension,
                min_tier=micro.min_tier,
                downgrade_variant=None,
                half_life=micro.half_life,
                teacher_prep_level=micro.teacher_prep_level,
                misconceptions=micro.misconceptions,
                evidence_of_mastery=micro.evidence_of_mastery,
                estimated_minutes=micro.estimated_minutes,
                source_refs=tuple(resolved),
                confidence=None,
            )
            identities[(unit.unit_id, micro.slug)] = node.id
            prepared.append((record, plan, unit, micro, parent, node))
    # Invalid citations cannot create valid graph nodes or dangling prerequisite chains.
    changed = True
    while changed:
        changed = False
        for _, _, unit, micro, _, _node in prepared:
            key = (unit.unit_id, micro.slug)
            if key in identities and any(
                (unit.unit_id, p) not in identities for p in micro.prerequisites
            ):
                del identities[key]
                quarantined.append(
                    {
                        "unit_id": unit.unit_id,
                        "slug": micro.slug,
                        "reason": "unresolved prerequisite after evidence rejection",
                        "version": version,
                    }
                )
                changed = True
    for record, plan, unit, micro, parent, node in prepared:
        if (unit.unit_id, micro.slug) not in identities:
            continue
        node_refs = node.source_refs
        prerequisites = tuple(
            PrerequisiteRef(id=identities[(unit.unit_id, p)], type="PREREQUISITE")
            for p in micro.prerequisites
        )
        node = MicroSkill.model_validate({**node.model_dump(), "prerequisites": prerequisites})
        if node.id in micros and micros[node.id] != node:
            raise ValueError(
                "same intrinsic micro-skill has conflicting definitions; explicit merge required"
            )
        micros[node.id] = node
        for prereq in prerequisites:
            edge = Edge(
                source=prereq.id,
                target=node.id,
                type="PREREQUISITE",
                justification="Proposed instructional prerequisite, not empirically validated: "
                + micro.pedagogical_rationale,
                source_refs=node_refs,
            )
            edges[edge.id] = edge
        edge = Edge(
            source=node.id,
            target=parent.id,
            type="REFINES",
            justification="Observable performance refines the high-level skill-map goal.",
            source_refs=node_refs,
        )
        edges[edge.id] = edge
        sources = [
            {
                **r.model_dump(mode="json"),
                "page": evidence.paragraphs[r.paragraph_id][1]["original"]["page"],
                "section": evidence.paragraphs[r.paragraph_id][1]["original"]["section"],
                "dom_path": evidence.paragraphs[r.paragraph_id][1]["original"]["dom_path"],
                "url": str(
                    next(a.document.url for a in evidence.records if a.document.id == r.document_id)
                ),
            }
            for r in node_refs
        ]
        row = {
            k: v
            for k, v in micro.model_dump(mode="json").items()
            if k not in {"evidence_quotes", "prerequisites"}
        }
        row.update(
            {
                "grade": unit.grade,
                "unit_id": unit.unit_id,
                "unit_title": plan.title,
                "generated_unit_title": unit.title,
                "publication_changes": [
                    r.rationale
                    for r in curation.removed_prerequisites
                    if version != "v1" and r.unit_id == unit.unit_id and r.skill_slug == micro.slug
                ],
                "micro_skill_id": node.id,
                "parent_skill_id": parent.id,
                "progression": plan.progression,
                "prerequisite_ids": [p.id for p in prerequisites],
                "sources": sources,
                "confidence": None,
                "review_status": "awaiting_independent_human_review",
                "dependency_status": "instructional_hypothesis",
                "selected_version": record.get("selected_version", "revised")
                if version != "v1"
                else "initial",
                "safety_notes": unit.safety_notes,
                "unit_limitations": list(unit.limitations),
                "outstanding_review_issues": (
                    record.get("final_review", {}) if version != "v1" else record["review"]
                ).get("issues", []),
            }
        )
        rows.append(row)
    source_ids = tuple(sorted({r.document_id for m in micros.values() for r in m.source_refs}))
    graph = GraphSnapshot(
        version=version,
        skills=tuple(skills.values()),
        micro_skills=tuple(micros.values()),
        edges=tuple(edges.values()),
        source_ids=source_ids,
    )
    return graph, rows, identities, quarantined


def build_release(root: Path, evidence: EvidenceIndex) -> dict[str, Any]:
    inputs = root / "data/processed/curriculum_experiment"
    records = [json.loads(p.read_text()) for p in sorted(inputs.glob("g*.json"))]
    plans = tuple(
        UnitPlan.model_validate(u)
        for u in json.loads((root / "config/curriculum_map.json").read_text())["units"]
    )
    if {p.id for p in plans} != {r["plan"]["id"] for r in records} or len(records) != len(plans):
        raise ValueError("release requires every declared unit exactly once")
    expected_plans = {p.id: p for p in plans}
    for record in records:
        if UnitPlan.model_validate(record["plan"]) != expected_plans[record["plan"]["id"]]:
            raise ValueError("proposal plan differs from the release map")
        verify_record(record, root / "data/processed/llm_responses", evidence)
    config = json.loads((root / "config/release_scenarios.json").read_text())
    scenario = ReleaseScenario.model_validate(config["primary"])
    curation_path = root / "config/curriculum_curation.json"
    curation = (
        CurriculumCuration.model_validate_json(curation_path.read_bytes())
        if curation_path.exists()
        else CurriculumCuration()
    )
    if any(r.unit_id not in expected_plans for r in curation.removed_prerequisites):
        raise ValueError("curation refers to an undeclared unit")
    snapshots = {}
    summaries = []
    for version in ("v1", "v2"):
        graph, rows, ids, quarantine = compile_graph(records, version, evidence, curation)
        output = root / "data/processed/curriculum_releases" / version
        output.mkdir(parents=True, exist_ok=True)
        for name in ("curriculum_proposal.html", "budget_ledger.jsonl", "sensitivity.json"):
            (output / name).unlink(missing_ok=True)
        snapshots[version] = graph
        write_json(output / "graph.json", graph)
        (output / "nodes.jsonl").write_text(
            "".join(canonical_json(n) + "\n" for n in (*graph.skills, *graph.micro_skills))
        )
        (output / "edges.jsonl").write_text("".join(canonical_json(e) + "\n" for e in graph.edges))
        (output / "curriculum.jsonl").write_text("".join(canonical_json(r) + "\n" for r in rows))
        write_json(output / "quarantine.json", quarantine)
        units = tuple(
            TeachingUnit.model_validate(
                r["initial" if version == "v1" else r.get("selected_version", "revised")]
            )
            for r in records
        )
        (output / "model_candidate_units.jsonl").write_text(
            "".join(canonical_json(u) + "\n" for u in units)
        )
        if version != "v1":
            units = tuple(curate(u, curation) for u in units)
        (output / "candidate_units.jsonl").write_text(
            "".join(canonical_json(u) + "\n" for u in units)
        )
        allocation: dict[str, Any]
        if quarantine or len(ids) != sum(len(u.micro_skills) for u in units):
            # The first snapshot may contain invalid drafts. It stays auditable, never schedulable.
            allocation = {
                "status": "blocked_by_invalid_evidence",
                "diagnostics": quarantine,
                "deployable": False,
            }
        else:
            allocation = schedule(units, plans, ids, scenario)
            (output / "curriculum_proposal.html").write_text(
                render_proposal(version, rows, allocation)
            )
        write_json(output / "schedule.json", allocation)
        if allocation.get("ledgers"):
            (output / "budget_ledger.jsonl").write_text(
                "".join(
                    canonical_json(row) + "\n"
                    for ledger in allocation["ledgers"]
                    for row in ledger["rows"]
                )
            )
        with (output / "curriculum_proposal.csv").open("w", newline="") as stream:
            fields = list(rows[0]) if rows else ["grade", "micro_skill_id"]
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: canonical_json(value) if isinstance(value, (list, dict)) else value
                        for key, value in row.items()
                    }
                )
        external = [
            {
                "micro_skill_id": r["micro_skill_id"],
                "grade": r["grade"],
                "requirement": requirement,
                "status": "unresolved_not_an_asserted_graph_edge",
            }
            for r in rows
            for requirement in r["external_knowledge"]
        ]
        write_json(output / "external_requirements.json", external)
        if version == "v2" and allocation.get("ledgers"):
            sensitivity = []
            for minutes in config["sensitivity_minutes"]:
                candidate = schedule(
                    units,
                    plans,
                    ids,
                    ReleaseScenario.model_validate(
                        {**scenario.model_dump(), "minutes_per_grade": minutes}
                    ),
                )
                sensitivity.append(
                    {
                        "minutes_per_grade": minutes,
                        "status": candidate["status"],
                        "grades": candidate["grades"],
                        "diagnostics": candidate["diagnostics"],
                    }
                )
            write_json(output / "sensitivity.json", sensitivity)
        summary = {
            "version": version,
            "micro_skills": len(graph.micro_skills),
            "edges": len(graph.edges),
            "quarantined": len(quarantine),
            "unresolved_external_requirements": len(external),
            "schedule_status": allocation["status"],
            "human_validated": False,
            "calibrated_confidence": False,
            "slices": [
                {
                    "grade": g,
                    "strand": strand,
                    "micro_skills": sum(r["grade"] == g and r["strand"] == strand for r in rows),
                }
                for g in range(2, 7)
                for strand in sorted({p.strand.value for p in plans})
            ],
        }
        write_json(output / "evaluation.json", summary)
        summaries.append(summary)
    comparison = {
        "versions": summaries,
        "schema_version": "release-comparison/1.0",
        "units": [
            {
                "unit_id": r["plan"]["id"],
                "initial_hard_errors": r["initial_errors"],
                "revised_hard_errors": r["final_errors"],
                "initial_critique": r["review"],
                "revised_critique": r.get("final_review"),
                "selected_version": r.get("selected_version", "revised"),
                "format_repairs": r.get("format_repairs", []),
            }
            for r in records
        ],
        "limitations": [
            "Critiques are uncalibrated model judgments, not independent human validation.",
            "Prerequisite necessity and proposed grade placement remain instructional hypotheses.",
            "Missing current Salvadoran Computer Science anchor prevents certification of vertical alignment.",
            "Host hours and specific equipment require local verification before deployment.",
        ],
    }
    release_root = root / "data/processed/curriculum_releases"
    progression_links = []
    for strand in sorted({p.strand for p in plans}):
        chain = sorted((p for p in plans if p.strand == strand), key=lambda p: p.grade)
        for earlier, later in zip(chain[:-1], chain[1:], strict=True):
            progression_links.append(
                {
                    "from_unit": earlier.id,
                    "to_unit": later.id,
                    "from_grade": earlier.grade,
                    "to_grade": later.grade,
                    "strand": strand.value,
                    "rationale": later.progression,
                    "status": "instructional_progression_hypothesis_not_a_necessary_prerequisite",
                }
            )
    write_json(release_root / "progression_links.json", progression_links)
    write_json(release_root / "curation.json", curation)
    write_json(release_root / "comparison.json", comparison)
    write_json(release_root / "graph_diff.json", diff_graphs(snapshots["v1"], snapshots["v2"]))
    provenance = build_provenance(root / "data")
    # Generated release files are outputs, not producer code or input modifications.
    source_root = Path(__file__).resolve().parents[3]
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            ".",
            ":!data/processed/curriculum_releases",
        ],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
    )
    provenance["git_dirty"] = bool(status.stdout.strip()) if status.returncode == 0 else None
    provenance["dirty_scope"] = "producer files; generated curriculum release directory excluded"
    commit = provenance["commit"]
    inputs_sha = {
        str(p.relative_to(root)): byte_digest(p.read_bytes()) for p in sorted(inputs.glob("*.json"))
    }
    for p in (
        root / "config/curriculum_map.json",
        root / "config/release_scenarios.json",
        root / "uv.lock",
    ):
        inputs_sha[str(p.relative_to(root))] = byte_digest(p.read_bytes())
    for relative in (
        "config/curriculum_map_initial.json",
        "config/curriculum_curation.json",
        "evaluation/release_editorial_review.yaml",
        "evaluation/curriculum_release_review.yaml",
    ):
        p = root / relative
        if p.exists():
            inputs_sha[relative] = byte_digest(p.read_bytes())
    response_hashes = {
        key: byte_digest((root / "data/processed/llm_responses" / (key + ".json")).read_bytes())
        for key in sorted({key for r in records for key in r["request_keys"]})
    }
    artifacts = {
        str(p.relative_to(release_root)): byte_digest(p.read_bytes())
        for p in sorted(release_root.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    manifest = {
        "schema_version": "curriculum-release/1.0",
        "pipeline_commit": commit,
        "provenance": provenance,
        "seed": 0,
        "status": "revised_research_proposal",
        "adoption_ready": False,
        "inputs_sha256": inputs_sha,
        "artifacts_sha256": artifacts,
        "response_keys": sorted(response_hashes),
        "response_records_sha256": response_hashes,
        "source_manifest_sha256": byte_digest((root / "data/manifests/corpus.jsonl").read_bytes()),
    }
    write_json(release_root / "manifest.json", manifest)
    return manifest
