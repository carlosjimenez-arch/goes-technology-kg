import pytest
from hypothesis import given
from hypothesis import strategies as st

from goes_tech_kg.corpus.chunk import chunk_document
from goes_tech_kg.corpus.normalize import normalize, original_range
from goes_tech_kg.schemas.corpus import Paragraph, ParsedDocument


def paragraph(text, id="p1", unit="activity"):
    return Paragraph(id=id, text=text, page=1, section="Procedure", unit_id=unit)


@given(st.text(min_size=1, max_size=400))
def test_normalization_maps_every_character(text):
    p = normalize(paragraph(text))
    expected = text.replace("\r\n", "\n").replace("\r", "\n")
    assert p.text == expected
    for i, c in enumerate(p.text):
        start, end = original_range(p, i, i + 1)
        assert text[start:end].replace("\r\n", "\n").replace("\r", "\n") == c
    start, end = original_range(p, 0, len(p.text))
    assert text[start:end] == text


@given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=8))
def test_chunking_preserves_all_normalized_characters(texts):
    parsed = ParsedDocument(
        document_id="s1",
        document_sha256="a" * 64,
        paragraphs=tuple(paragraph(t, f"p{i}", f"unit-{i // 3}") for i, t in enumerate(texts)),
    )
    chunks = chunk_document(parsed)
    assert "".join(c.text for c in chunks) == "".join(
        t.replace("\r\n", "\n").replace("\r", "\n") for t in texts
    )
    assert [p.original.id for c in chunks for p in c.paragraphs] == [
        p.id for p in parsed.paragraphs
    ]
    assert len(chunks) == (len(texts) + 2) // 3


def test_invalid_evidence_offset_fails():
    p = normalize(paragraph("abc"))
    for start, end in [(-1, 1), (0, 4), (2, 2)]:
        with pytest.raises(ValueError):
            original_range(p, start, end)
