"""PDF paragraphs and complete HTML activity bodies with stable source locators."""

import io
import re

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from pypdf import PdfReader

from goes_tech_kg.schemas.base import byte_digest, stable_id
from goes_tech_kg.schemas.corpus import Paragraph, ParsedDocument, SourceDocument

PARSER_VERSION = "parse/1.1"


def _paragraph(
    sha: str,
    text: str,
    ordinal: int,
    unit: str,
    page: int | None = None,
    section: str = "Document",
    dom_path: str | None = None,
) -> Paragraph:
    return Paragraph(
        id=stable_id(
            "paragraph",
            f"p{ordinal}",
            {"source": sha, "page": page, "path": dom_path, "ordinal": ordinal, "text": text},
        ),
        text=text,
        page=page,
        section=section,
        dom_path=dom_path,
        unit_id=unit,
    )


def parse_pdf(document: SourceDocument, payload: bytes) -> ParsedDocument:
    sha = byte_digest(payload)
    if document.reviewed_sha256 and sha != document.reviewed_sha256:
        raise ValueError("reviewed PDF content changed")
    reader = PdfReader(io.BytesIO(payload))
    paragraphs: list[Paragraph] = []
    boundaries = document.unit_boundaries
    if boundaries and boundaries[-1].start_page > len(reader.pages):
        raise ValueError("unit boundary outside PDF")
    for page_no, page in enumerate(reader.pages, 1):
        if document.text_extraction == "plain":
            text = page.extract_text()
        else:
            text = page.extract_text(extraction_mode="layout", layout_mode_strip_rotated=False)
            if not text.strip():
                text = page.extract_text()

        if page_no in document.page_text_overrides:
            if text.strip():
                raise ValueError("reviewed textless page now has extractable text")
            text = document.page_text_overrides[page_no]
            if not text:
                continue
        if not text.strip():
            # Fail closed for textless/scanned pages rather than silently dropping evidence.
            raise ValueError(
                f"PDF page {page_no} has no extractable text; OCR/manual review required"
            )
        active = [u for u in boundaries if u.start_page <= page_no]
        unit = active[-1].label if active else "Front matter"
        # Delimiters belong to the preceding paragraph, preserving extracted text exactly.
        parts = re.findall(r".*?(?:\n[ \t]*\n|\Z)", text, flags=re.S)
        for part in parts:
            if part:
                paragraphs.append(_paragraph(sha, part, len(paragraphs) + 1, unit, page_no, unit))
    return ParsedDocument(
        document_id=document.id, document_sha256=sha, paragraphs=tuple(paragraphs)
    )


def _dom_path(tag: Tag) -> str:
    parts = []
    current: Tag | None = tag
    while current is not None and current.name != "[document]":
        siblings = current.find_previous_siblings(current.name)
        parts.append(f"{current.name}:nth-of-type({len(siblings) + 1})")
        current = current.parent if isinstance(current.parent, Tag) else None
    return " > ".join(reversed(parts))


def parse_html(document: SourceDocument, payload: bytes) -> ParsedDocument:
    sha = byte_digest(payload)
    soup = BeautifulSoup(payload, "html.parser")
    root = soup.select_one(document.root_selector)
    if root is None:
        raise ValueError(f"required content selector not found: {document.root_selector}")
    for tag in root.select("script,style,nav,footer,noscript"):
        tag.decompose()
    paragraphs: list[Paragraph] = []
    section = document.title
    unit = document.title
    # Entire selected activity is one unit; subheadings never split its procedure from assessment.
    for element in root.descendants:
        if isinstance(element, Tag) and element.name in {"h1", "h2", "h3", "h4"}:
            section = element.get_text(" ", strip=True) or section
        if not isinstance(element, NavigableString) or isinstance(element, Comment):
            continue
        text = str(element)
        if not text:
            continue
        parent = element.parent
        if not isinstance(parent, Tag):
            continue
        paragraphs.append(
            _paragraph(
                sha,
                text,
                len(paragraphs) + 1,
                unit,
                section=section,
                dom_path=_dom_path(parent),
            )
        )
    if not any(p.text.strip() for p in paragraphs):
        raise ValueError("empty HTML activity")
    return ParsedDocument(
        document_id=document.id, document_sha256=sha, paragraphs=tuple(paragraphs)
    )


def parse(document: SourceDocument, payload: bytes) -> ParsedDocument:
    return (
        parse_pdf(document, payload) if document.format == "pdf" else parse_html(document, payload)
    )
