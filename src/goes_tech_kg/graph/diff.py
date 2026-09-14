"""Semantic diff by intrinsic identity; order-only changes disappear."""

from goes_tech_kg.schemas.base import digest
from goes_tech_kg.schemas.graph import EntityDelta, GraphDiff, GraphSnapshot, Modification
from goes_tech_kg.schemas.skills import Edge, MicroSkill, Skill


def _delta(
    before: tuple[Skill | MicroSkill | Edge, ...], after: tuple[Skill | MicroSkill | Edge, ...]
) -> EntityDelta:
    old = {x.id: x.model_dump(mode="json") for x in before}
    new = {x.id: x.model_dump(mode="json") for x in after}
    modified = []
    for key in sorted(old.keys() & new.keys()):
        fields = tuple(
            sorted(
                k for k in old[key].keys() | new[key].keys() if old[key].get(k) != new[key].get(k)
            )
        )
        if fields:
            modified.append(
                Modification(id=key, changed_fields=fields, before=old[key], after=new[key])
            )
    return EntityDelta(
        added=tuple(new[k] for k in sorted(new.keys() - old.keys())),
        removed=tuple(old[k] for k in sorted(old.keys() - new.keys())),
        modified=tuple(modified),
    )


def diff_graphs(before: GraphSnapshot, after: GraphSnapshot) -> GraphDiff:
    return GraphDiff(
        before_sha256=digest(before),
        after_sha256=digest(after),
        nodes=_delta((*before.skills, *before.micro_skills), (*after.skills, *after.micro_skills)),
        edges=_delta(before.edges, after.edges),
        changed_rules=before.dependency_rules != after.dependency_rules,
        added_source_ids=tuple(sorted(set(after.source_ids) - set(before.source_ids))),
        removed_source_ids=tuple(sorted(set(before.source_ids) - set(after.source_ids))),
    )
