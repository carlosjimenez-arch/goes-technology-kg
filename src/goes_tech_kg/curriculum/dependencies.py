"""Report external-grade conflicts without moving or allocating skills."""

from goes_tech_kg.schemas.graph import GraphSnapshot
from goes_tech_kg.schemas.skills import SkillPlacement


def cross_subject_conflicts(
    graph: GraphSnapshot, placements: tuple[SkillPlacement, ...]
) -> list[dict[str, object]]:
    nodes = {m.id: m for m in graph.micro_skills}
    if len({p.micro_skill_id for p in placements}) != len(placements):
        raise ValueError("duplicate placement")
    result = []
    for p in placements:
        if p.micro_skill_id not in nodes:
            raise ValueError("unknown micro-skill placement")
        for dep in nodes[p.micro_skill_id].cross_subject_deps:
            if dep.available_grade > p.grade:
                result.append(
                    {
                        "micro_skill_id": p.micro_skill_id,
                        "requested_grade": p.grade,
                        "dependency": dep.qualified_id,
                        "available_grade": dep.available_grade,
                    }
                )
    return sorted(result, key=lambda x: str(x["micro_skill_id"]) + str(x["dependency"]))
