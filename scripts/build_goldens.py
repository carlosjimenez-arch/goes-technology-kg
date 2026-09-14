"""Create bounded real-source fixtures; source and excerpt hashes are retained."""

import base64
import io
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader, PdfWriter

from goes_tech_kg.corpus.chunk import chunk_document
from goes_tech_kg.corpus.parse import parse
from goes_tech_kg.corpus.pipeline import read_manifest
from goes_tech_kg.schemas.base import byte_digest, canonical_json
from goes_tech_kg.schemas.corpus import SourceDocument

root = Path(__file__).resolve().parents[1]
records = {r.document.slug: r for r in read_manifest(root / "data/manifests/corpus.jsonl")}
for slug in ["eng-dt", "eng-computing", "unplugged-rocket"]:
    record = records[slug]
    body = (root / "data/raw/corpus" / record.sha256).read_bytes()
    document = record.document.model_dump(mode="json")
    document["unit_boundaries"] = []
    if slug == "eng-dt":
        # A one-page crop of actual PDF page 2; never generated curriculum text.
        reader = PdfReader(io.BytesIO(body))
        writer = PdfWriter()
        writer.add_page(reader.pages[1])
        stream = io.BytesIO()
        writer.write(stream)
        payload = stream.getvalue()
        extraction = {
            "source_pages": [2],
            "method": "pypdf page-object crop; fixture page 1 maps to original page 2",
        }
        phrases = ["Subject content", "Design", "Make"]
    else:
        soup = BeautifulSoup(body, "html.parser")
        if slug == "eng-computing":
            heading = soup.find(id="key-stage-1")
            assert heading is not None
            chosen = [heading]
            for sibling in heading.next_siblings:
                if getattr(sibling, "name", None) == "h2":
                    break
                chosen.append(sibling)
            html = "".join(str(x) for x in chosen)
            extraction = {
                "dom_selector": "#key-stage-1",
                "method": "Heading and complete following siblings through next key-stage heading; inert main wrapper added.",
            }
            phrases = ["algorithms", "debug simple programs"]
        else:
            # Complete lesson body (one activity), excluding global page navigation/footer.
            selected = soup.select_one("#content-container")
            assert selected is not None
            html = str(selected)
            extraction = {
                "dom_selector": "#content-container",
                "method": "Complete activity subtree; HTML serialization, no invented teaching content.",
            }
            phrases = ["Sending a rocket to Mars", "Algorithm", "Lesson"]
        payload = ("<main>" + html + "</main>").encode()
        document["root_selector"] = "main"
    # Excerpt bytes have a separate digest; no claim that a crop hashes to its source.
    document["reviewed_sha256"] = None
    document["page_text_overrides"] = {}
    document["page_review_reasons"] = {}
    doc = SourceDocument.model_validate(document)
    parsed = parse(doc, payload)
    chunks = chunk_document(parsed)
    text = "".join(c.text for c in chunks)
    # These phrases are human-selected anchors; fail rather than auto-accepting extraction changes.
    for phrase in phrases:
        if phrase not in text:
            raise ValueError(f"{slug}: expected phrase missing: {phrase}")
    golden = {
        "schema_version": "real-source-excerpt/1.0",
        "source_url": str(record.document.url),
        "source_sha256": record.sha256,
        "acquired_at": record.acquired_at.isoformat(),
        "license": "CC-BY-SA-4.0" if slug.startswith("unplugged") else "OGL-3.0",
        "attribution": "CS Unplugged, University of Canterbury"
        if slug.startswith("unplugged")
        else "Department for Education, Crown copyright",
        "extraction": extraction,
        "excerpt_sha256": byte_digest(payload),
        "document": doc.model_dump(mode="json"),
        "payload_base64": base64.b64encode(payload).decode(),
        "paragraph_count": len(parsed.paragraphs),
        "chunk_count": len(chunks),
        "normalized_sha256": byte_digest(text.encode()),
        "required_phrases": phrases,
    }
    (root / "tests/golden" / f"{slug}.json").write_text(canonical_json(golden) + "\n")
    print(slug, len(payload), "bytes", len(parsed.paragraphs), "paragraphs")
