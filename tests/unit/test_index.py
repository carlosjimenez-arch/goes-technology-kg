import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from goes_tech_kg.corpus.index import VectorIndex


@given(st.permutations([0, 1, 2]))
def test_index_order_and_ties(order):
    # Same real index logic for each ordering; no substitute implementations.
    vectors = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    ids = ("b", "a", "c")
    index = VectorIndex(tuple(ids[i] for i in order), vectors[list(order)])
    assert index.search(np.array([1, 0], dtype=np.float32), 2) == [("a", 1.0), ("b", 1.0)]


def test_index_round_trip_and_corruption(tmp_path):
    index = VectorIndex(("b", "a"), np.array([[0, 1], [1, 0]], dtype=np.float32))
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.search(np.array([1, 0], dtype=np.float32))[0] == ("a", 1.0)
    (tmp_path / "vectors.f32").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        VectorIndex.load(tmp_path)
