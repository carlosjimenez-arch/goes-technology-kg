"""Explicit publication edits preserve original model proposals and reject stale rules."""

from goes_tech_kg.schemas.release import CurriculumCuration, TeachingUnit


def curate(unit: TeachingUnit, rules: CurriculumCuration) -> TeachingUnit:
    body = unit.model_dump(mode="json")
    by_slug = {m["slug"]: m for m in body["micro_skills"]}
    for rule in rules.removed_prerequisites:
        if rule.unit_id != unit.unit_id:
            continue
        if (
            rule.skill_slug not in by_slug
            or rule.prerequisite_slug not in by_slug[rule.skill_slug]["prerequisites"]
        ):
            raise ValueError("stale prerequisite curation rule")
        by_slug[rule.skill_slug]["prerequisites"].remove(rule.prerequisite_slug)
    return TeachingUnit.model_validate(body)
