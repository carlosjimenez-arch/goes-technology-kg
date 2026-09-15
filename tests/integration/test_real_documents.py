import hashlib

from goes_tech_kg.corpus.chunk import chunk_document
from goes_tech_kg.corpus.normalize import original_range
from goes_tech_kg.corpus.parse import parse


def test_real_document_goldens(golden_documents):
    for document, payload, golden in golden_documents:
        assert hashlib.sha256(payload).hexdigest() == golden["excerpt_sha256"]
        parsed = parse(document, payload)
        chunks = chunk_document(parsed)
        assert len(parsed.paragraphs) == golden["paragraph_count"]
        text = "".join(c.text for c in chunks)
        assert hashlib.sha256(text.encode()).hexdigest() == golden["normalized_sha256"]
        for phrase in golden["required_phrases"]:
            assert phrase in text
        assert len(chunks) == golden["chunk_count"]
        for c in chunks:
            for p in c.paragraphs:
                if p.text:
                    left, right = original_range(p, 0, len(p.text))
                    assert p.original.text[left:right] == p.original.text


def test_plain_extraction_mode_reads_columns_in_stream_order(golden_documents):
    from goes_tech_kg.corpus.parse import parse
    from goes_tech_kg.schemas.corpus import SourceDocument

    document, payload, _ = next(row for row in golden_documents if row[0].format == "pdf")
    plain = SourceDocument.model_validate(
        {**document.model_dump(mode="json"), "text_extraction": "plain"}
    )
    layout_text = "".join(p.text for p in parse(document, payload).paragraphs)
    plain_text = "".join(p.text for p in parse(plain, payload).paragraphs)
    assert plain_text.strip() and plain_text != layout_text
    # Both modes keep every word of the page; only spacing and order differ.
    assert set(layout_text.split()) >= {"Design", "Make"} <= set(plain_text.split())
