"""Decision 0011 primary metrics: align a decomposition to a golden record and score it."""

import re

from goes_tech_kg.eval.prompt_metrics import indicator_ids
from goes_tech_kg.schemas.decomposition import DecompositionOutput, ProposedMicroSkill
from goes_tech_kg.schemas.golden import GoldenRecord, GoldenScore

PARAGRAPH_KEY = re.compile(r"¶paragraph-(p\d+)-")


def system_keys(micro: ProposedMicroSkill) -> frozenset[str]:
    """Evidence keys of a proposed micro-skill: indicator ids and paragraph keys it cites."""
    keys: set[str] = set()
    for quote in micro.evidence_quotes:
        keys |= indicator_ids(quote.quote)
        keys |= set(PARAGRAPH_KEY.findall(quote.locator_hint))
    return frozenset(keys)


def align(output: DecompositionOutput, golden: GoldenRecord) -> dict[str, frozenset[str]]:
    """Map each system slug to every golden id sharing an evidence key (many-to-many).

    A system micro-skill that merges two indicators legitimately covers two golden items;
    one-to-one alignment would punish the merge that decision 0013 rule 17 asks for.
    """
    mapping: dict[str, frozenset[str]] = {}
    for micro in output.micro_skills:
        keys = system_keys(micro)
        mapping[micro.slug] = frozenset(
            item.id for item in golden.micro_skills if keys & frozenset(item.evidence_keys)
        )
    return mapping


def score(
    output: DecompositionOutput,
    golden: GoldenRecord,
    system: str = "system",
    model: str = "none",
) -> GoldenScore:
    """Decision 0011 primary metrics for one decomposition against its reference.

    Recall counts golden items some system skill cites; precision counts system skills that
    cite any golden item. A prerequisite edge is correct when any aligned pair is golden,
    so a legitimately merged micro-skill is not punished for covering two references.
    """
    mapping = align(output, golden)
    matched_golden = {g for ids in mapping.values() for g in ids}
    golden_ids = {g.id for g in golden.micro_skills}
    system_count = len(output.micro_skills)
    decomposition_recall = len(matched_golden) / len(golden_ids)
    decomposition_precision = (
        sum(1 for ids in mapping.values() if ids) / system_count if system_count else 0.0
    )
    # A system edge is a set of candidate golden pairs; it is correct if any pair is golden.
    system_edges: list[frozenset[tuple[str, str]]] = []
    for m in output.micro_skills:
        for p in m.prerequisites:
            pairs = frozenset(
                (a, b) for a in mapping.get(p, frozenset()) for b in mapping[m.slug] if a != b
            )
            if pairs:
                system_edges.append(pairs)
    golden_edges = golden.edge_set
    correct = sum(1 for pairs in system_edges if pairs & golden_edges)
    covered_golden_edges = {e for pairs in system_edges for e in pairs & golden_edges}
    edge_precision = correct / len(system_edges) if system_edges else None
    edge_recall = len(covered_golden_edges) / len(golden_edges) if golden_edges else None
    if edge_precision is None or edge_recall is None or (edge_precision + edge_recall) == 0:
        edge_f1 = 0.0 if system_edges or golden_edges else None
    else:
        edge_f1 = 2 * edge_precision * edge_recall / (edge_precision + edge_recall)
    by_id = {g.id: g for g in golden.micro_skills}
    aligned = [(m, by_id[g]) for m in output.micro_skills for g in sorted(mapping[m.slug])]
    cognitive_agreement = (
        sum(m.cognitive_domain == g.cognitive_domain for m, g in aligned) / len(aligned)
        if aligned
        else None
    )
    tier_agreement = (
        sum(m.min_tier == g.min_tier for m, g in aligned) / len(aligned) if aligned else None
    )
    strand_agreement = (
        sum(m.strand == g.strand for m, g in aligned) / len(aligned) if aligned else None
    )
    return GoldenScore(
        case_id=golden.case_id,
        grade=golden.grade,
        system=system,
        model=model,
        golden_micro_skills=len(golden_ids),
        system_micro_skills=system_count,
        decomposition_recall=decomposition_recall,
        decomposition_precision=decomposition_precision,
        edge_precision=edge_precision,
        edge_recall=edge_recall,
        edge_f1=edge_f1,
        system_edges=len(system_edges),
        golden_edges=len(golden_edges),
        cognitive_agreement=cognitive_agreement,
        tier_agreement=tier_agreement,
        strand_agreement=strand_agreement,
        unmatched_golden=tuple(sorted(golden_ids - matched_golden)),
    )


def official_baseline(golden: GoldenRecord) -> GoldenScore:
    """B0: the official programme covers a golden item when the item cites an indicator id."""
    covered = [
        g for g in golden.micro_skills if any(indicator_ids(k + ".") for k in g.evidence_keys)
    ]
    indicators = {k for g in covered for k in g.evidence_keys if indicator_ids(k + ".")}
    return GoldenScore(
        case_id=golden.case_id,
        grade=golden.grade,
        system="B0-official-programme",
        model="none",
        golden_micro_skills=len(golden.micro_skills),
        decomposition_recall=len(covered) / len(golden.micro_skills),
        indicators=len(indicators),
        granularity_golden_per_indicator=(len(covered) / len(indicators) if indicators else None),
        edge_recall=0.0 if golden.edge_set else None,
        golden_edges=len(golden.edge_set),
        note="The programme states indicators, not prerequisite edges, cognitive domains or tiers.",
    )
