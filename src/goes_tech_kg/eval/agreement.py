"""Inter-rater agreement statistics for judge validation (decision 0011, section 4)."""

from collections import Counter
from collections.abc import Sequence
from itertools import combinations

import numpy as np

Label = str | int | float


def cohen_kappa(a: Sequence[Label], b: Sequence[Label]) -> float:
    """Chance-corrected agreement for two raters on nominal labels."""
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equal-length, nonempty ratings")
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / len(a)
    labels = set(a) | set(b)
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum(counts_a[label] * counts_b[label] for label in labels) / len(a) ** 2
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def weighted_kappa(a: Sequence[int], b: Sequence[int], levels: Sequence[int]) -> float:
    """Quadratic-weighted kappa for two raters on an ordinal scale with the given levels."""
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equal-length, nonempty ratings")
    index = {level: i for i, level in enumerate(levels)}
    k = len(levels)
    observed = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        observed[index[x], index[y]] += 1
    observed /= observed.sum()
    marginal_a, marginal_b = observed.sum(axis=1), observed.sum(axis=0)
    expected = np.outer(marginal_a, marginal_b)
    weights = np.array([[((i - j) / (k - 1)) ** 2 for j in range(k)] for i in range(k)])
    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return 1.0
    return 1.0 - float((weights * observed).sum()) / denominator


def krippendorff_alpha(ratings: Sequence[Sequence[Label | None]], metric: str = "nominal") -> float:
    """Krippendorff's alpha over raters x items; None marks a missing rating.

    metric is "nominal" (any label) or "ordinal"/"interval" (numeric labels). Items rated by
    fewer than two raters are excluded, which is the statistic's standard treatment of
    missing data.
    """
    if metric not in {"nominal", "ordinal", "interval"}:
        raise ValueError("metric must be nominal, ordinal or interval")
    items = [
        [value for value in column if value is not None] for column in zip(*ratings, strict=True)
    ]
    items = [values for values in items if len(values) >= 2]
    if not items:
        raise ValueError("alpha needs at least one item with two or more ratings")
    values_all = [value for values in items for value in values]
    n = len(values_all)
    counts = Counter(values_all)
    levels = sorted(counts, key=lambda v: (str(type(v)), v))

    def delta(u: Label, v: Label) -> float:
        if metric == "nominal":
            return 0.0 if u == v else 1.0
        if metric == "interval":
            return (float(u) - float(v)) ** 2
        # ordinal: squared difference of cumulative frequency midpoints
        i, j = sorted((levels.index(u), levels.index(v)))
        if i == j:
            return 0.0
        between = sum(counts[levels[g]] for g in range(i, j + 1))
        return (between - (counts[levels[i]] + counts[levels[j]]) / 2) ** 2

    observed = 0.0
    for values in items:
        m = len(values)
        observed += sum(delta(u, v) for u, v in combinations(values, 2)) * 2 / (m - 1)
    observed /= n
    expected = sum(delta(u, v) for u, v in combinations(values_all, 2)) * 2 / (n - 1)
    expected /= n
    if expected == 0.0:
        return 1.0
    return 1.0 - observed / expected


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation for continuous judge scores against a human mean."""
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("spearman needs at least three paired scores")
    ranks_a = _ranks(a)
    ranks_b = _ranks(b)
    if np.std(ranks_a) == 0 or np.std(ranks_b) == 0:
        raise ValueError("constant ratings have no rank correlation")
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = array.argsort(kind="stable")
    ranks = np.empty(len(array), dtype=float)
    i = 0
    while i < len(array):
        j = i
        while j + 1 < len(array) and array[order[j + 1]] == array[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def activation(statistic: float, gate: float = 0.6, advise: float = 0.4) -> str:
    """Decision 0011 activation status from a judge-human agreement statistic."""
    if statistic >= gate:
        return "gating"
    if statistic >= advise:
        return "advisory"
    return "deactivated"
