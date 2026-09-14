"""Dynamic micro-skill confidence: raw evidence score, golden-fitted calibration, verification."""

from collections.abc import Iterable

import numpy as np

from goes_tech_kg.schemas.base import digest
from goes_tech_kg.schemas.calibration import (
    VERDICT_VALUE,
    CalibrationBin,
    CalibrationSample,
    CalibrationTable,
    ConfidenceBasis,
)
from goes_tech_kg.schemas.graph import GraphSnapshot
from goes_tech_kg.schemas.skills import MicroSkill


def raw_score(basis: ConfidenceBasis) -> float | None:
    """Weighted judge agreement, discounted by citation support and revisions.

    Returns None when no calibrated judge carries weight: absence of evidence is not evidence.
    """
    weights = np.asarray([v.weight for v in basis.judge_verdicts], dtype=np.float64)
    if weights.size == 0 or float(weights.sum()) == 0.0:
        return None
    values = np.asarray([VERDICT_VALUE[v.verdict] for v in basis.judge_verdicts], dtype=np.float64)
    judged = float(np.dot(weights, values) / weights.sum())
    # Each accepted revision means an earlier version was wrong; discount geometrically.
    revision_factor = 1.0 / (1.0 + basis.revision_count)
    return min(1.0, max(0.0, judged * basis.citation_support * revision_factor))


def calibrated_confidence(basis: ConfidenceBasis, table: CalibrationTable | None) -> float | None:
    """Human review is ground truth for the item; otherwise map the raw score through the table."""
    if basis.human_review in ("accepted", "corrected"):
        return 1.0
    if basis.human_review == "rejected":
        return 0.0
    score = raw_score(basis)
    if score is None or table is None:
        return None
    return table.apply(score)


def fit_calibration(
    samples: Iterable[CalibrationSample], golden_sha256: str, min_support: int = 5
) -> CalibrationTable:
    """Pool-adjacent-violators isotonic regression of human acceptance on raw score.

    Deterministic: samples are sorted by (raw_score, label) before pooling. Bins with fewer
    than min_support items are merged into their right neighbour so no probability rests on
    a handful of items.
    """
    rows = sorted(samples, key=lambda s: (s.raw_score, s.human_accepted))
    if len(rows) < min_support:
        raise ValueError("insufficient golden samples for calibration")
    if any(s.annotation_method != "human_unaided" for s in rows):
        raise ValueError("calibration accepts only unaided human labels")
    # Each block: [sum of labels, count, upper raw score]
    blocks: list[list[float]] = [[float(s.human_accepted), 1.0, s.raw_score] for s in rows]
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(blocks) - 1:
            left, right = blocks[i], blocks[i + 1]
            if left[0] / left[1] > right[0] / right[1]:
                blocks[i : i + 2] = [[left[0] + right[0], left[1] + right[1], right[2]]]
                merged = True
            else:
                i += 1
    i = 0
    while i < len(blocks):
        if blocks[i][1] < min_support and len(blocks) > 1:
            j = i + 1 if i + 1 < len(blocks) else i - 1
            lo, hi = sorted((i, j))
            a, b = blocks[lo], blocks[hi]
            blocks[lo : hi + 1] = [[a[0] + b[0], a[1] + b[1], b[2]]]
            i = 0
            continue
        i += 1
    blocks[-1][2] = 1.0
    bins = tuple(
        CalibrationBin(raw_upper=b[2], probability=b[0] / b[1], support=int(b[1])) for b in blocks
    )
    return CalibrationTable(golden_sha256=golden_sha256, bins=bins)


def _table_digest(
    micro: MicroSkill, value: float | None, table: CalibrationTable | None
) -> str | None:
    if value is None or table is None or micro.human_reviewed:
        return None
    return digest(table)


def attach_confidence(micro: MicroSkill, table: CalibrationTable | None) -> MicroSkill:
    """Recompute and stamp confidence from the recorded basis; identity does not change."""
    if micro.confidence_basis is None:
        raise ValueError("micro-skill has no confidence basis to calibrate")
    value = calibrated_confidence(micro.confidence_basis, table)
    data = micro.model_dump(mode="json")
    data.update({"confidence": value, "calibration_sha256": _table_digest(micro, value, table)})
    return MicroSkill.model_validate(data)


def verify_confidence(snapshot: GraphSnapshot, table: CalibrationTable | None) -> list[str]:
    """IDs whose stored confidence does not reproduce from basis and table. Empty means consistent."""
    failures = []
    for micro in snapshot.micro_skills:
        if micro.confidence is None and micro.confidence_basis is None:
            continue
        if micro.confidence_basis is None:
            failures.append(micro.id)
            continue
        value = calibrated_confidence(micro.confidence_basis, table)
        if value != micro.confidence or micro.calibration_sha256 != _table_digest(
            micro, value, table
        ):
            failures.append(micro.id)
    return sorted(failures)
