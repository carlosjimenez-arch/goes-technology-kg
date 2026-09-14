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
