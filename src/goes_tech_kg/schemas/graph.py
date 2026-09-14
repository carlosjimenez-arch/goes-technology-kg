"""Snapshot integrity and typed dependency rules; no inference or sequencing solver."""

from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from goes_tech_kg.schemas.base import Contract, Text
from goes_tech_kg.schemas.skills import Edge, EdgeType, HalfLife, MicroSkill, Skill


class DependencyRule(Contract):
    target: Text
    any_of: Annotated[tuple[tuple[Text, ...], ...], Field(min_length=1)]
    justification: Text

    @model_validator(mode="after")
    def clauses(self) -> Self:
        if any(not c or len(c) != len(set(c)) or self.target in c for c in self.any_of):
            raise ValueError("empty, duplicate or self prerequisite clause")
        normalized = tuple(sorted(tuple(sorted(c)) for c in self.any_of))
        if len(set(normalized)) != len(normalized):
            raise ValueError("duplicate alternative")
        object.__setattr__(self, "any_of", normalized)
        return self


class GraphSnapshot(Contract):
    schema_version: Literal["tech-graph/1.0"] = "tech-graph/1.0"
    version: Text
    skills: tuple[Skill, ...] = ()
    micro_skills: tuple[MicroSkill, ...] = ()
    edges: tuple[Edge, ...] = ()
    source_ids: tuple[Text, ...] = ()
    dependency_rules: tuple[DependencyRule, ...] = ()

    @model_validator(mode="after")
    def integrity(self) -> Self:
        for field in ("skills", "micro_skills", "edges"):
            values = getattr(self, field)
            if len({v.id for v in values}) != len(values):
                raise ValueError(f"duplicate {field} IDs")
            object.__setattr__(self, field, tuple(sorted(values, key=lambda v: v.id)))
        nodes = {n.id for n in self.skills} | {n.id for n in self.micro_skills}
        if len(nodes) != len(self.skills) + len(self.micro_skills):
            raise ValueError("node identity collision")
        if any(m.half_life == HalfLife.VOLATILE for m in self.micro_skills):
            raise ValueError("VOLATILE belongs exclusively to tool_bindings, never the graph")
        parents = {s.id for s in self.skills}
        external = {d.qualified_id for m in self.micro_skills for d in m.cross_subject_deps}
        for m in self.micro_skills:
            if m.parent_skill_id not in parents:
                raise ValueError("dangling parent")
        for e in self.edges:
            if e.target not in nodes or e.source not in (
                external if e.type == EdgeType.CROSS_SUBJECT_PREREQUISITE else nodes
            ):
                raise ValueError("dangling or incorrectly qualified edge endpoint")
        refs = {r.document_id for x in self.micro_skills for r in x.source_refs}
        refs.update(r.document_id for edge in self.edges for r in edge.source_refs)
        refs.update(
            r.document_id
            for m in self.micro_skills
            for d in m.cross_subject_deps
            for r in d.source_refs
        )
        if not refs.issubset(self.source_ids):
            raise ValueError("unregistered source reference")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("duplicate source ID")
        object.__setattr__(self, "source_ids", tuple(sorted(self.source_ids)))
        self._inline_integrity()
        self._strict_acyclicity(nodes)
        for rule in self.dependency_rules:
            if rule.target not in nodes or any(p not in nodes for c in rule.any_of for p in c):
                raise ValueError("dangling dependency rule endpoint")
        object.__setattr__(
            self, "dependency_rules", tuple(sorted(self.dependency_rules, key=lambda r: r.target))
        )
        return self

    def _inline_integrity(self) -> None:
        for m in self.micro_skills:
            expected = {
                (e.source, e.type)
                for e in self.edges
                if e.target == m.id and e.type in (EdgeType.PREREQUISITE, EdgeType.CO_REQUISITE)
            }
            expected |= {
                (e.target, e.type)
                for e in self.edges
                if e.source == m.id and e.type == EdgeType.CO_REQUISITE
            }
            if {(p.id, p.type) for p in m.prerequisites} != expected:
                raise ValueError("inline prerequisites must exactly mirror edges")
            actual_external = {
                e.source
                for e in self.edges
                if e.target == m.id and e.type == EdgeType.CROSS_SUBJECT_PREREQUISITE
            }
            if actual_external != {d.qualified_id for d in m.cross_subject_deps}:
                raise ValueError("cross-subject declarations must mirror edges")

    def _strict_acyclicity(self, nodes: set[str]) -> None:
        positions = {key: i for i, key in enumerate(sorted(nodes))}
        pairs = [
            (positions[e.source], positions[e.target])
            for e in self.edges
            if e.type == EdgeType.PREREQUISITE
        ]
        if not pairs:
            return
        rows, cols = zip(*pairs, strict=True)
        matrix = csr_matrix((np.ones(len(pairs)), (rows, cols)), shape=(len(nodes), len(nodes)))
        _, labels = connected_components(matrix, directed=True, connection="strong")
        if np.max(np.bincount(labels)) > 1:
            raise ValueError("strict prerequisite cycle")


class Modification(Contract):
    id: Text
    changed_fields: tuple[Text, ...]
    before: dict[str, object]
    after: dict[str, object]


class EntityDelta(Contract):
    added: tuple[dict[str, object], ...] = ()
    removed: tuple[dict[str, object], ...] = ()
    modified: tuple[Modification, ...] = ()


class GraphDiff(Contract):
    before_sha256: Text
    after_sha256: Text
    nodes: EntityDelta
    edges: EntityDelta
    changed_rules: bool
    added_source_ids: tuple[str, ...]
    removed_source_ids: tuple[str, ...]
