import pytest
from hypothesis import given
from hypothesis import strategies as st

from goes_tech_kg.eval import agreement


def test_cohen_kappa_known_values():
    assert agreement.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    # Classic textbook table: 20 agree on yes, 15 on no, 5 and 10 disagree -> kappa 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert agreement.cohen_kappa(a, b) == pytest.approx(0.4)
    assert agreement.cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
    with pytest.raises(ValueError):
        agreement.cohen_kappa(["a"], ["a", "b"])


def test_weighted_kappa_penalizes_distance():
    levels = [1, 2, 3, 4, 5]
    close = agreement.weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 4], levels)
    far = agreement.weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 1], levels)
    assert 1.0 > close > far
    assert agreement.weighted_kappa([1, 3, 5], [1, 3, 5], levels) == 1.0


def test_krippendorff_alpha_nominal_and_ordinal_and_missing():
    perfect = [["a", "b", "c", "a"], ["a", "b", "c", "a"]]
    assert agreement.krippendorff_alpha(perfect) == pytest.approx(1.0)
    disagreeing = [["a", "b", "a", "b"], ["b", "a", "b", "a"]]
    assert agreement.krippendorff_alpha(disagreeing) < 0
    with_missing = [["a", "b", None, "a"], ["a", "b", "c", None], ["a", None, "c", "a"]]
    assert 0 < agreement.krippendorff_alpha(with_missing) <= 1
    ordinal = [[1, 2, 3, 4], [1, 2, 3, 5]]
    assert agreement.krippendorff_alpha(ordinal, "ordinal") > agreement.krippendorff_alpha(
        [[1, 2, 3, 4], [1, 2, 3, 1]], "ordinal"
    )
    with pytest.raises(ValueError):
        agreement.krippendorff_alpha([[None, None], [None, None]])
    with pytest.raises(ValueError):
        agreement.krippendorff_alpha(perfect, "ratio")


@given(st.lists(st.sampled_from(["accept", "revise", "reject"]), min_size=2, max_size=40))
def test_kappa_and_alpha_are_one_for_identical_raters(labels):
    assert agreement.cohen_kappa(labels, labels) == pytest.approx(1.0)
    assert agreement.krippendorff_alpha([labels, labels]) == pytest.approx(1.0)


def test_spearman_and_activation():
    assert agreement.spearman([0.1, 0.5, 0.9], [0.2, 0.6, 0.8]) == pytest.approx(1.0)
    assert agreement.spearman([0.1, 0.5, 0.9], [0.9, 0.5, 0.1]) == pytest.approx(-1.0)
    assert agreement.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        agreement.spearman([1, 1, 1], [1, 2, 3])
    assert agreement.activation(0.75) == "gating"
    assert agreement.activation(0.5) == "advisory"
    assert agreement.activation(0.1) == "deactivated"
