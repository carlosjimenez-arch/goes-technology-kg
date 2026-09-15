"""Acquisition, licensing and evidence contracts with explicit stage state."""

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field, HttpUrl, model_validator

from goes_tech_kg.schemas.base import Contract, Digest, Grade, Text, stable_id


class UnitBoundary(Contract):
    start_page: Annotated[int, Field(strict=True, ge=1)]
    label: Text
    grades: tuple[Grade, ...] = ()


class SourceDocument(Contract):
    schema_version: Literal["tech-source/1.0"] = "tech-source/1.0"
    id: str = ""
    slug: Text
    title: Text
    url: HttpUrl
    publisher: Text
    country: Text
    role: Literal[
        "foreign_curriculum", "international_framework", "salvadoran_baseline", "supplementary"
    ]
    # Degenerate-loop barrier: pipeline output is never admissible as corpus evidence.
    provenance: Literal["external", "generated"]
    grades: tuple[Grade, ...]
    alignment_rationale: Text
    format: Literal["pdf", "html"]
    license: Text | None
    admission_basis: Literal["official_publication", "open_license", "unverified"]
    license_evidence_url: HttpUrl | None
    license_evidence: Text | None
    reviewed_at: datetime
    root_selector: Text = "main"
    # layout keeps horizontal positions (interleaves table columns line by line);
    # plain follows the content stream, which reads multi-column tables column by column.
    text_extraction: Literal["layout", "plain"] = "layout"
    unit_boundaries: tuple[UnitBoundary, ...] = ()
    required_baseline: bool = False
    reviewed_sha256: Digest | None = None
    page_text_overrides: dict[int, str] = {}
    page_review_reasons: dict[int, Text] = {}

    @model_validator(mode="after")
    def identity(self) -> Self:
        expected = stable_id("source", self.slug, {"url": str(self.url)})
        if self.id and self.id != expected:
            raise ValueError("source identity mismatch")
        object.__setattr__(self, "id", expected)
        if len(set(self.grades)) != len(self.grades):
            raise ValueError("duplicate grade")
        if set(self.page_text_overrides) != set(self.page_review_reasons) or (
            self.page_text_overrides and not self.reviewed_sha256
        ):
            raise ValueError("page overrides require digest-pinned review reasons")
        starts = [u.start_page for u in self.unit_boundaries]
        if starts != sorted(set(starts)):
            raise ValueError("unit boundaries must be unique and ordered")
        return self


class LicenseDecision(Contract):
    document_id: Text
    status: Literal["accepted", "rejected"]
    reason: Text
    policy_sha256: Digest
    evidence_url: HttpUrl | None
    reviewed_at: datetime


class Acquisition(Contract):
    document: SourceDocument
    license_decision: LicenseDecision
    status: Literal["accepted", "rejected", "fetch_failed"]
    sha256: Digest | None = None
    byte_count: Annotated[int, Field(strict=True, ge=0)] | None = None
    acquired_at: datetime | None = None
    final_url: HttpUrl | None = None
    error: Text | None = None

    @model_validator(mode="after")
    def state(self) -> Self:
        if self.document.id != self.license_decision.document_id:
            raise ValueError("gate document mismatch")
        if self.status == "accepted" and (
            self.license_decision.status != "accepted"
            or not self.sha256
            or not self.byte_count
            or not self.acquired_at
            or not self.final_url
        ):
            raise ValueError(
                "accepted source requires successful license gate and complete acquisition"
            )
        if self.status != "accepted" and (self.sha256 is not None or not self.error):
            raise ValueError("failed source needs reason and cannot claim acquired digest")
        return self


class Paragraph(Contract):
    id: Text
    text: str
    page: int | None = None
    section: Text
    dom_path: Text | None = None
    unit_id: Text

    @model_validator(mode="after")
    def locator(self) -> Self:
        if self.page is None and self.dom_path is None:
            raise ValueError("paragraph requires PDF page or HTML DOM path")
        return self


class ParsedDocument(Contract):
    document_id: Text
    document_sha256: Digest
    parser_version: Literal["parse/1.1"] = "parse/1.1"
    paragraphs: tuple[Paragraph, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if not self.paragraphs or len({p.id for p in self.paragraphs}) != len(self.paragraphs):
            raise ValueError("empty parse or duplicate paragraph IDs")
        return self


class OffsetSpan(Contract):
    normalized_start: int
    normalized_end: int
    original_start: int
    original_end: int


class NormalizedParagraph(Contract):
    original: Paragraph
    text: str
    offset_map: tuple[OffsetSpan, ...]

    @model_validator(mode="after")
    def exact_map(self) -> Self:
        n = o = 0
        for span in self.offset_map:
            if (
                span.normalized_start != n
                or span.original_start != o
                or span.normalized_end <= n
                or span.original_end <= o
            ):
                raise ValueError("offset map must be a contiguous nonempty partition")
            expected = (
                self.original.text[o : span.original_end].replace("\r\n", "\n").replace("\r", "\n")
            )
            if expected != self.text[n : span.normalized_end]:
                raise ValueError("offset map does not reconstruct original text")
            n, o = span.normalized_end, span.original_end
        if n != len(self.text) or o != len(self.original.text):
            raise ValueError("offset map loses text")
        return self


class Chunk(Contract):
    id: Text
    document_id: Text
    document_sha256: Digest
    unit_id: Text
    paragraphs: tuple[NormalizedParagraph, ...]
    text: str

    @model_validator(mode="after")
    def complete(self) -> Self:
        if not self.paragraphs or self.text != "".join(p.text for p in self.paragraphs):
            raise ValueError("chunk must preserve every normalized paragraph character")
        if any(p.original.unit_id != self.unit_id for p in self.paragraphs):
            raise ValueError("chunk crosses curricular unit")
        return self
