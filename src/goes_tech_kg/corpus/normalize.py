"""Coordinate-preserving CRLF normalization with compact linear offset runs."""

import re

from goes_tech_kg.schemas.corpus import NormalizedParagraph, OffsetSpan, Paragraph


def normalize(paragraph: Paragraph) -> NormalizedParagraph:
    text = paragraph.text
    parts = []
    spans = []
    i = 0
    j = 0
    for match in re.finditer(r"\r\n?", text):
        if match.start() > i:
            part = text[i : match.start()]
            parts.append(part)
            spans.append(
                OffsetSpan(
                    normalized_start=j,
                    normalized_end=j + len(part),
                    original_start=i,
                    original_end=match.start(),
                )
            )
            j += len(part)
        parts.append("\n")
        spans.append(
            OffsetSpan(
                normalized_start=j,
                normalized_end=j + 1,
                original_start=match.start(),
                original_end=match.end(),
            )
        )
        j += 1
        i = match.end()
    if i < len(text):
        parts.append(text[i:])
        spans.append(
            OffsetSpan(
                normalized_start=j,
                normalized_end=j + len(text) - i,
                original_start=i,
                original_end=len(text),
            )
        )
    return NormalizedParagraph(original=paragraph, text="".join(parts), offset_map=tuple(spans))


def original_range(paragraph: NormalizedParagraph, start: int, end: int) -> tuple[int, int]:
    if not 0 <= start < end <= len(paragraph.text):
        raise ValueError("invalid normalized interval")
    selected = [
        s for s in paragraph.offset_map if s.normalized_start < end and s.normalized_end > start
    ]
    first, last = selected[0], selected[-1]
    left = first.original_start + start - first.normalized_start
    right = last.original_end - (last.normalized_end - end)
    return left, right
