import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from goes_tech_kg.eval.calibration import (
    attach_confidence,
    calibrated_confidence,
    fit_calibration,
    raw_score,
    verify_confidence,
)
from goes_tech_kg.schemas.base import digest
from goes_tech_kg.schemas.calibration import CalibrationSample, CalibrationTable, ConfidenceBasis
from goes_tech_kg.schemas.graph import GraphSnapshot

GOLDEN = "c" * 64


def verdict(judge, verdict, agreement, sha="d"):
    return {
        "judge_id": judge,
        "judge_llm_version": "provider/model@2026-09",
        "verdict": verdict,
        "agreement_with_humans": agreement,
        "response_sha256": sha * 64,
    }


def basis(**updates):
    data = {
        "judge_verdicts": [verdict("j1", "accept", 0.8), verdict("j2", "reject", 0.7, "e")],
        "citation_support": 1.0,
        "revision_count": 0,
    }
    data.update(updates)
    return ConfidenceBasis.model_validate(data)


def table(bins=((0.5, 0.2, 10), (1.0, 0.9, 10))):
    return CalibrationTable(
        golden_sha256=GOLDEN,
        bins=[{"raw_upper": u, "probability": p, "support": s} for u, p, s in bins],
    )


def test_raw_score_weights_judges_by_human_agreement_and_discounts_revisions():
    # weights: j1 (0.8-0.4)/0.6 = 2/3, j2 (0.7-0.4)/0.6 = 1/2 -> (2/3*1 + 1/2*0)/(7/6) = 4/7
    assert raw_score(basis()) == pytest.approx(4 / 7)
    assert raw_score(basis(revision_count=1)) == pytest.approx(2 / 7)
    assert raw_score(basis(citation_support=0.5)) == pytest.approx(2 / 7)


def test_absence_of_calibrated_judges_yields_no_confidence():
    assert raw_score(basis(judge_verdicts=[])) is None
    uncalibrated = basis(judge_verdicts=[verdict("j1", "accept", None)])
    assert raw_score(uncalibrated) is None
    weak = basis(judge_verdicts=[verdict("j1", "accept", 0.4)])
    assert raw_score(weak) is None
    assert calibrated_confidence(basis(), None) is None


def test_human_review_is_ground_truth_for_the_item():
    assert calibrated_confidence(basis(human_review="accepted"), None) == 1.0
    assert calibrated_confidence(basis(human_review="corrected"), None) == 1.0
    assert calibrated_confidence(basis(human_review="rejected"), table()) == 0.0


def test_calibration_table_applies_bins_and_rejects_nonmonotone():
    t = table()
    assert calibrated_confidence(basis(), t) == 0.9
    assert calibrated_confidence(basis(revision_count=1), t) == 0.2
    with pytest.raises(ValidationError, match="nondecreasing"):
        table(bins=((0.5, 0.9, 10), (1.0, 0.2, 10)))
    with pytest.raises(ValidationError, match="cover raw score 1.0"):
        table(bins=((0.5, 0.2, 10),))


def test_fit_calibration_known_values_and_human_only():
    samples = [
        CalibrationSample(raw_score=r, human_accepted=a, annotation_method="human_unaided")
        for r, a in [(0.1, False)] * 5 + [(0.3, True)] + [(0.3, False)] * 4 + [(0.9, True)] * 5
    ]
    t = fit_calibration(samples, GOLDEN, min_support=5)
    assert [b.probability for b in t.bins] == pytest.approx([0.0, 0.2, 1.0])
    assert [b.raw_upper for b in t.bins] == [0.1, 0.3, 1.0]
    with pytest.raises(ValidationError):
        CalibrationSample(raw_score=0.5, human_accepted=True, annotation_method="llm_assisted")
    with pytest.raises(ValueError, match="insufficient"):
        fit_calibration(samples[:3], GOLDEN)


@given(
    st.lists(st.tuples(st.floats(0, 1, allow_nan=False), st.booleans()), min_size=5, max_size=60)
)
def test_fitted_table_is_monotone_and_permutation_invariant(pairs):
    samples = [
        CalibrationSample(raw_score=r, human_accepted=a, annotation_method="human_unaided")
        for r, a in pairs
    ]
    first = fit_calibration(samples, GOLDEN)
    second = fit_calibration(list(reversed(samples)), GOLDEN)
    assert first == second
    probabilities = [b.probability for b in first.bins]
    assert probabilities == sorted(probabilities)
    assert sum(b.support for b in first.bins) == len(pairs)


def test_micro_skill_confidence_cannot_be_authored(micro_factory):
    with pytest.raises(ValidationError, match="confidence basis"):
        micro_factory(confidence=0.8)
    with pytest.raises(ValidationError, match="calibration table digest"):
        micro_factory(confidence=0.8, confidence_basis=basis().model_dump(mode="json"))
    with pytest.raises(ValidationError):
        micro_factory(confidence=float("nan"), confidence_basis=basis().model_dump(mode="json"))


def test_attach_and_verify_confidence_round_trip(skill, micro_factory):
    t = table()
    pending = micro_factory(confidence_basis=basis().model_dump(mode="json"))
    assert pending.confidence is None
    stamped = attach_confidence(pending, t)
    assert stamped.confidence == 0.9 and stamped.calibration_sha256 == digest(t)
    assert stamped.id == pending.id
    reviewed = attach_confidence(
        micro_factory(
            slug="reviewed",
            confidence_basis=basis(human_review="accepted").model_dump(mode="json"),
        ),
        t,
    )
    assert reviewed.confidence == 1.0 and reviewed.calibration_sha256 is None

    def snapshot(*micros):
        return GraphSnapshot(
            version="v1", skills=(skill,), micro_skills=micros, source_ids=("source-fixture",)
        )

    unstamped = micro_factory(slug="pending", confidence_basis=basis().model_dump(mode="json"))
    assert verify_confidence(snapshot(stamped, reviewed), t) == []
    # A basis that could have been calibrated but was not is stale, not neutral.
    assert verify_confidence(snapshot(unstamped), t) == [unstamped.id]
    assert verify_confidence(snapshot(unstamped), None) == []
    other = table(bins=((0.5, 0.1, 10), (1.0, 0.8, 10)))
    assert verify_confidence(snapshot(stamped), other) == [stamped.id]
    tampered = stamped.model_validate({**stamped.model_dump(mode="json"), "confidence": 0.99})
    assert verify_confidence(snapshot(tampered), t) == [tampered.id]
    with pytest.raises(ValueError, match="no confidence basis"):
        attach_confidence(micro_factory(slug="bare"), t)
