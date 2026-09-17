"""Model-free measurements of a decomposition: no judge, no model, only text and contracts.

Metrics that need the prompt context go through `ContextIndex`, which derives the normalized
text, its tokens, its locators and its indicator numbers once per context instead of once per
metric. Metrics that only need the decomposition itself stay pure functions.
"""

import re
import unicodedata
from functools import cached_property

from goes_tech_kg.schemas.decomposition import DecompositionOutput, ProposedMicroSkill
from goes_tech_kg.schemas.experiment import DecompositionMetrics
from goes_tech_kg.schemas.skills import CognitiveDomain, Tier

#: Verbs that name a mental state rather than an observable performance.
NON_OBSERVABLE_VERBS = frozenset(
    {
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
)
#: Ordinal positions used to average a cognitive-domain distribution.
COGNITIVE_ORDINAL = {
    CognitiveDomain.KNOWING: 0,
    CognitiveDomain.APPLYING: 1,
    CognitiveDomain.REASONING: 2,
}
_INDICATOR = re.compile(r"\b(\d{1,2})\s*\.\s*(\d{1,2})\s*\.")
_LOCATOR = re.compile(r"\[[^\]\n]+\]")
_WORD = re.compile(r"[^\W_]+")
_HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")
_LETTER_SPACED = re.compile(r"\b(?:\w\s){2,}\w\b")


def normalize(text: str) -> str:
    """Casefold, NFKC-normalize and collapse whitespace so comparisons ignore layout."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def repair_layout(text: str) -> str:
    """Undo two PDF extraction artifacts: hyphenated line breaks and letter-spaced words."""
    return _LETTER_SPACED.sub(
        lambda m: m.group(0).replace(" ", ""), _HYPHEN_BREAK.sub(r"\1\2", text)
    )


def tokens(text: str) -> tuple[str, ...]:
    """Word tokens of a normalized, layout-repaired text."""
    return tuple(_WORD.findall(repair_layout(normalize(text))))


def indicator_ids(text: str) -> frozenset[str]:
    """Numbered achievement indicators (for example 5.3) appearing in a text."""
    return frozenset(f"{a}.{b}" for a, b in _INDICATOR.findall(text))


def signature(micro: ProposedMicroSkill) -> tuple[str, str]:
    """Identity of a proposed micro-skill for comparison: its verb and its knowledge object."""
    return (normalize(micro.observable_verb), normalize(micro.knowledge_object))


def _quotes(output: DecompositionOutput) -> list[str]:
    return [q.quote for m in output.micro_skills for q in m.evidence_quotes]


class ContextIndex:
    """Derived views of one assembled prompt context, computed once and reused by every metric.

    The Salvadoran programme interleaves table columns line by line, so a faithful quote of one
    column is not a verbatim substring of the context. This class therefore offers two
    strictness levels: `quote_exactness` for literal presence and `quote_support` for an
    ordered token subsequence inside a bounded window.
    """

    def __init__(self, context: str) -> None:
        self.text = context

    @cached_property
    def normalized(self) -> str:
        """The whole context, normalized for substring comparison."""
        return normalize(self.text)

    @cached_property
    def tokens(self) -> tuple[str, ...]:
        """Layout-repaired word tokens of the whole context."""
        return tokens(self.text)

    @cached_property
    def locators(self) -> frozenset[str]:
        """Bracketed paragraph locators offered to the model, without their brackets."""
        return frozenset(x.strip("[]") for x in _LOCATOR.findall(self.text))

    @cached_property
    def indicators(self) -> frozenset[str]:
        """Numbered indicators present in the context."""
        return indicator_ids(self.text)

    @cached_property
    def _token_starts(self) -> dict[str, tuple[int, ...]]:
        starts: dict[str, list[int]] = {}
        for position, token in enumerate(self.tokens):
            starts.setdefault(token, []).append(position)
        return {token: tuple(positions) for token, positions in starts.items()}

    def supports(self, quote: str, slack: float = 1.0) -> bool:
        """True when every quote token appears in order inside a bounded context window.

        The window is (1 + slack) times the quote length, which accepts a column read across
        interleaved lines while still rejecting a sentence the context never contained.
        """
        words = tokens(quote)
        if not words:
            return False
        limit = int(len(words) * (1 + slack)) + 2
        for start in self._token_starts.get(words[0], ()):
            position = start
            end = min(len(self.tokens), start + limit)
            for word in words[1:]:
                while position + 1 < end and self.tokens[position + 1] != word:
                    position += 1
                position += 1
                if position >= end or self.tokens[position] != word:
                    break
            else:
                return True
        return False

    def contains_verbatim(self, quote: str) -> bool:
        """True when the quote appears literally, ignoring only whitespace and case."""
        return normalize(quote) in self.normalized

    def quote_exactness(self, output: DecompositionOutput) -> float | None:
        """Share of evidence quotes found verbatim in the context."""
        quotes = _quotes(output)
        if not quotes:
            return None
        return sum(self.contains_verbatim(q) for q in quotes) / len(quotes)

    def quote_support(self, output: DecompositionOutput) -> float | None:
        """Share of evidence quotes supported as ordered token subsequences of the context."""
        quotes = _quotes(output)
        if not quotes:
            return None
        return sum(self.supports(q) for q in quotes) / len(quotes)

    def unsupported_quotes(self, output: DecompositionOutput) -> list[tuple[str, str]]:
        """Micro-skill slug and quote for every quote the context does not support."""
        return [
            (m.slug, q.quote)
            for m in output.micro_skills
            for q in m.evidence_quotes
            if not self.supports(q.quote)
        ]

    def locator_exactness(self, output: DecompositionOutput) -> float | None:
        """Share of locator hints that name a locator the context actually offered."""
        hints = [q.locator_hint for m in output.micro_skills for q in m.evidence_quotes]
        if not hints:
            return None
        # Models sometimes drop the brackets; the locator itself is what must match.
        return sum(h.strip().strip("[]") in self.locators for h in hints) / len(hints)

    def indicator_coverage(
        self, output: DecompositionOutput, in_scope: frozenset[str] | None = None
    ) -> float | None:
        """Share of the context's numbered indicators quoted by some micro-skill.

        With `in_scope` (decision 0014) only Technology indicators count: Science dependencies
        and excluded indicators are neither required nor credited.
        """
        expected = self.indicators if in_scope is None else self.indicators & in_scope
        if not expected:
            return None
        quoted: set[str] = set()
        for m in output.micro_skills:
            for q in m.evidence_quotes:
                quoted |= indicator_ids(q.quote)
        return len(expected & quoted) / len(expected)


def observable_rate(output: DecompositionOutput) -> float | None:
    """Share of micro-skills whose verb names a performance rather than a mental state."""
    if not output.micro_skills:
        return None
    return sum(
        normalize(m.observable_verb).split(" ")[0] not in NON_OBSERVABLE_VERBS
        for m in output.micro_skills
    ) / len(output.micro_skills)


def t0_share(output: DecompositionOutput) -> float | None:
    """Share of micro-skills teachable and assessable without any device."""
    if not output.micro_skills:
        return None
    return sum(m.min_tier == Tier.T0 for m in output.micro_skills) / len(output.micro_skills)


def duplicate_rate(output: DecompositionOutput) -> float | None:
    """Share of micro-skills that repeat another one's verb and knowledge object."""
    if not output.micro_skills:
        return None
    keys = [signature(m) for m in output.micro_skills]
    return 1 - len(set(keys)) / len(keys)


def mean_cognitive_level(output: DecompositionOutput) -> float | None:
    """Average cognitive demand on the knowing/applying/reasoning ordinal scale."""
    if not output.micro_skills:
        return None
    return sum(COGNITIVE_ORDINAL[m.cognitive_domain] for m in output.micro_skills) / len(
        output.micro_skills
    )


def mean_minutes(output: DecompositionOutput) -> float | None:
    """Average estimated classroom minutes per micro-skill."""
    if not output.micro_skills:
        return None
    return sum(m.estimated_minutes for m in output.micro_skills) / len(output.micro_skills)


def jaccard(a: DecompositionOutput, b: DecompositionOutput) -> float:
    """Overlap of two decompositions by verb and knowledge object; brittle to rewording."""
    left = {signature(m) for m in a.micro_skills}
    right = {signature(m) for m in b.micro_skills}
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def indicator_set_jaccard(a: DecompositionOutput, b: DecompositionOutput) -> float | None:
    """Overlap of the indicators two decompositions quote; None when neither quotes any."""

    def quoted(output: DecompositionOutput) -> set[str]:
        return {
            i
            for m in output.micro_skills
            for q in m.evidence_quotes
            for i in indicator_ids(q.quote)
        }

    left, right = quoted(a), quoted(b)
    if not left and not right:
        return None
    return len(left & right) / len(left | right)


def forbidden_term_hits(output: DecompositionOutput, terms: tuple[str, ...]) -> int:
    """How many forbidden terms appear in the decomposition's own prose (injection probe)."""
    text = normalize(
        " ".join(
            f"{m.statement} {m.knowledge_object} {m.evidence_of_mastery}"
            for m in output.micro_skills
        )
    )
    return sum(normalize(t) in text for t in terms)


def summarize(
    output: DecompositionOutput, context: ContextIndex, in_scope: frozenset[str] | None = None
) -> DecompositionMetrics:
    """Every model-free metric for one decomposition, as a validated contract."""
    return DecompositionMetrics(
        micro_skill_count=len(output.micro_skills),
        quote_exactness=context.quote_exactness(output),
        quote_support=context.quote_support(output),
        locator_exactness=context.locator_exactness(output),
        observable_rate=observable_rate(output),
        indicator_coverage=context.indicator_coverage(output),
        scoped_indicator_coverage=(
            context.indicator_coverage(output, in_scope) if in_scope is not None else None
        ),
        t0_share=t0_share(output),
        duplicate_rate=duplicate_rate(output),
        mean_cognitive_level=mean_cognitive_level(output),
        prerequisite_count=sum(len(m.prerequisites) for m in output.micro_skills),
        mean_minutes=mean_minutes(output),
    )
