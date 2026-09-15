"""Automatic, model-free metrics over a decomposition output and its prompt context."""

import re
import unicodedata

from goes_tech_kg.schemas.decomposition import DecompositionOutput, ProposedMicroSkill
from goes_tech_kg.schemas.skills import CognitiveDomain, Tier

NON_OBSERVABLE_VERBS = {
    "conoce",
    "conocer",
    "comprende",
    "comprender",
    "aprende",
    "aprender",
    "sabe",
    "saber",
    "entiende",
    "entender",
    "valora",
    "valorar",
    "reflexiona",
    "reflexionar",
    "reconoce",
}
COGNITIVE_ORDINAL = {
    CognitiveDomain.KNOWING: 0,
    CognitiveDomain.APPLYING: 1,
    CognitiveDomain.REASONING: 2,
}
INDICATOR = re.compile(r"\b(\d{1,2})\s*\.\s*(\d{1,2})\s*\.")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def quote_exactness(output: DecompositionOutput, context: str) -> float | None:
    """Share of evidence quotes found verbatim (whitespace-insensitive) in the context."""
    haystack = normalize(context)
    quotes = [q.quote for m in output.micro_skills for q in m.evidence_quotes]
    if not quotes:
        return None
    return sum(normalize(q) in haystack for q in quotes) / len(quotes)


_HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")
_LETTER_SPACED = re.compile(r"\b(?:\w\s){2,}\w\b")


def _repair_layout(text: str) -> str:
    """Undo two PDF extraction artifacts: hyphenated line breaks and letter-spaced words."""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _LETTER_SPACED.sub(lambda m: m.group(0).replace(" ", ""), text)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", _repair_layout(normalize(text)))


def quote_supported(quote: str, context_tokens: list[str], slack: float = 1.0) -> bool:
    """True when every quote token appears in order inside a bounded context window.

    Source PDFs interleave table columns line by line, so a faithful reading of one column is
    not a verbatim substring. An ordered subsequence within a window of (1 + slack) times the
    quote length accepts such reconstructions while rejecting invented sentences.
    """
    words = _tokens(quote)
    if not words:
        return False
    limit = int(len(words) * (1 + slack)) + 2
    starts = [i for i, tok in enumerate(context_tokens) if tok == words[0]]
    for start in starts:
        position = start
        end = min(len(context_tokens), start + limit)
        matched = True
        for word in words[1:]:
            while position + 1 < end and context_tokens[position + 1] != word:
                position += 1
            position += 1
            if position >= end or context_tokens[position] != word:
                matched = False
                break
        if matched:
            return True
    return False


def quote_support(output: DecompositionOutput, context: str) -> float | None:
    """Share of quotes supported as ordered token subsequences of the context."""
    tokens = _tokens(context)
    quotes = [q.quote for m in output.micro_skills for q in m.evidence_quotes]
    if not quotes:
        return None
    return sum(quote_supported(q, tokens) for q in quotes) / len(quotes)


def unsupported_quotes(output: DecompositionOutput, context: str) -> list[tuple[str, str]]:
    tokens = _tokens(context)
    return [
        (m.slug, q.quote)
        for m in output.micro_skills
        for q in m.evidence_quotes
        if not quote_supported(q.quote, tokens)
    ]


def locator_exactness(output: DecompositionOutput, context: str) -> float | None:
    locators = {x.strip("[]") for x in re.findall(r"\[[^\]\n]+\]", context)}
    hints = [q.locator_hint for m in output.micro_skills for q in m.evidence_quotes]
    if not hints:
        return None
    # Models sometimes drop the brackets; the locator itself is what must match.
    return sum(h.strip().strip("[]") in locators for h in hints) / len(hints)


def observable_rate(output: DecompositionOutput) -> float | None:
    if not output.micro_skills:
        return None
    return sum(
        normalize(m.observable_verb).split(" ")[0] not in NON_OBSERVABLE_VERBS
        for m in output.micro_skills
    ) / len(output.micro_skills)


def indicator_ids(context: str) -> set[str]:
    return {f"{a}.{b}" for a, b in INDICATOR.findall(context)}


def indicator_coverage(output: DecompositionOutput, context: str) -> float | None:
    """Share of numbered achievement indicators in the context quoted by some micro-skill."""
    ids = indicator_ids(context)
    if not ids:
        return None
    quoted: set[str] = set()
    for m in output.micro_skills:
        for q in m.evidence_quotes:
            quoted |= indicator_ids(q.quote)
    return len(ids & quoted) / len(ids)


def t0_share(output: DecompositionOutput) -> float | None:
    if not output.micro_skills:
        return None
    return sum(m.min_tier == Tier.T0 for m in output.micro_skills) / len(output.micro_skills)


def duplicate_rate(output: DecompositionOutput) -> float | None:
    if not output.micro_skills:
        return None
    keys = [signature(m) for m in output.micro_skills]
    return 1 - len(set(keys)) / len(keys)


def signature(micro: ProposedMicroSkill) -> tuple[str, str]:
    return (normalize(micro.observable_verb), normalize(micro.knowledge_object))


def mean_cognitive_level(output: DecompositionOutput) -> float | None:
    if not output.micro_skills:
        return None
    return sum(COGNITIVE_ORDINAL[m.cognitive_domain] for m in output.micro_skills) / len(
        output.micro_skills
    )


def jaccard(a: DecompositionOutput, b: DecompositionOutput) -> float:
    """Invariance measure between two decompositions using (verb, object) signatures."""
    left = {signature(m) for m in a.micro_skills}
    right = {signature(m) for m in b.micro_skills}
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def indicator_set_jaccard(a: DecompositionOutput, b: DecompositionOutput) -> float | None:
    """Invariance over which numbered indicators each decomposition quotes; None if neither quotes any."""

    def quoted(output: DecompositionOutput) -> set[str]:
        ids: set[str] = set()
        for m in output.micro_skills:
            for q in m.evidence_quotes:
                ids |= indicator_ids(q.quote)
        return ids

    left, right = quoted(a), quoted(b)
    if not left and not right:
        return None
    return len(left & right) / len(left | right)


def forbidden_term_hits(output: DecompositionOutput, terms: tuple[str, ...]) -> int:
    text = normalize(
        " ".join(
            f"{m.statement} {m.knowledge_object} {m.evidence_of_mastery}"
            for m in output.micro_skills
        )
    )
    return sum(normalize(t) in text for t in terms)


def summarize(output: DecompositionOutput, context: str) -> dict[str, float | int | None]:
    return {
        "micro_skill_count": len(output.micro_skills),
        "quote_exactness": quote_exactness(output, context),
        "quote_support": quote_support(output, context),
        "locator_exactness": locator_exactness(output, context),
        "observable_rate": observable_rate(output),
        "indicator_coverage": indicator_coverage(output, context),
        "t0_share": t0_share(output),
        "duplicate_rate": duplicate_rate(output),
        "mean_cognitive_level": mean_cognitive_level(output),
        "prerequisite_count": sum(len(m.prerequisites) for m in output.micro_skills),
        "mean_minutes": (
            sum(m.estimated_minutes for m in output.micro_skills) / len(output.micro_skills)
            if output.micro_skills
            else None
        ),
    }
