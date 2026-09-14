import numpy as np
import pytest

from goes_tech_kg.corpus.index import VectorIndex
from goes_tech_kg.corpus.parse import parse


@pytest.mark.benchmark
def test_parse_pdf(benchmark, golden_documents):
    doc, payload, _ = next(row for row in golden_documents if row[0].format == "pdf")
    parsed = benchmark(parse, doc, payload)
    assert parsed.paragraphs and parsed.paragraphs[0].page == 1


@pytest.mark.benchmark
def test_parse_html(benchmark, golden_documents):
    doc, payload, _ = next(row for row in golden_documents if row[0].format == "html")
    parsed = benchmark(parse, doc, payload)
    assert any(p.dom_path for p in parsed.paragraphs)


@pytest.mark.benchmark
def test_index_1000x768(benchmark):
    vectors = np.random.default_rng(0).normal(size=(1000, 768)).astype(np.float32)
    ids = tuple(f"chunk-{i:04d}" for i in range(1000))
    index = benchmark(VectorIndex, ids, vectors)
    assert index.vectors.shape == (1000, 768) and index.ids == ids
