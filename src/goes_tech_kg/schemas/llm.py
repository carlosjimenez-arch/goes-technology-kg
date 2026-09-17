"""Canonical LLM request and immutable, credential-free response records."""

from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from goes_tech_kg.schemas.base import Contract, Digest, Text, byte_digest, digest

Provider = Literal["vertex-ai"]


def provider_schema(contract: type[Contract]) -> dict[str, Any]:
    """The JSON schema sent to the provider, stripped of every description.

    Pydantic copies class docstrings into the schema as descriptions, and the schema is part
    of the request and therefore of the replay key. Stripping them keeps maintainer
    documentation free to change without invalidating recorded answers; wording meant for the
    model belongs in the versioned prompt, where it is measured.
    """
    stripped: dict[str, Any] = _without_descriptions(contract.model_json_schema())
    return stripped


def _without_descriptions(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _without_descriptions(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_without_descriptions(v) for v in node]
    return node


class GenerationSettings(Contract):
    """Provider settings that change the answer, and therefore the replay key."""

    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] = 0.0
    seed: Annotated[int, Field(strict=True, ge=0)] = 0
    max_output_tokens: Annotated[int, Field(strict=True, gt=0)] = 32768
    # Gemini 2.5 accepts a token budget; Gemini 3 accepts a level. Exactly one may be set.
    thinking_budget: Annotated[int, Field(strict=True, ge=0)] | None = None
    thinking_level: Literal["low", "medium", "high"] | None = None
    response_mime_type: Literal["application/json"] = "application/json"

    @model_validator(mode="after")
    def one_thinking_control(self) -> Self:
        if self.thinking_budget is not None and self.thinking_level is not None:
            raise ValueError("choose thinking_budget or thinking_level, not both")
        return self


class ContextRef(Contract):
    """Locator of the source text injected into a prompt; the text itself is never stored."""

    chunk_id: Text
    document_id: Text
    document_sha256: Digest
    paragraph_ids: Annotated[tuple[Text, ...], Field(min_length=1)]


class LLMRequest(Contract):
    """Everything that determines an answer, with the rendered prompt reduced to its digest.

    The rendered text is not stored because it embeds licensed corpus passages; the context
    is kept as locators instead, so a record can be committed while the source cannot.
    """

    schema_version: Literal["llm-request/1.0"] = "llm-request/1.0"
    provider: Provider = "vertex-ai"
    model: Text
    location: Text
    prompt_id: Text
    prompt_version: Text
    system_instruction: str
    # The rendered user content is hashed, not stored: it may embed licensed corpus text.
    user_content_sha256: Digest
    case_id: Text
    context_refs: tuple[ContextRef, ...] = ()
    output_schema_version: Text
    output_json_schema: dict[str, Any]
    settings: GenerationSettings = GenerationSettings()
    # Distinguishes deliberate repeated acquisitions of an otherwise identical request.
    replicate: Annotated[int, Field(strict=True, ge=0)] = 0
    cache_format_version: Literal["llm-cache/1.0"] = "llm-cache/1.0"

    @property
    def key(self) -> str:
        """Replay key: the digest of every field that can change the answer."""
        return digest(self)


class TokenUsage(Contract):
    """Tokens the provider reported, thinking counted separately from output."""

    prompt_tokens: Annotated[int, Field(strict=True, ge=0)]
    output_tokens: Annotated[int, Field(strict=True, ge=0)]
    thoughts_tokens: Annotated[int, Field(strict=True, ge=0)] = 0


class ResponseRecord(Contract):
    """One immutable provider answer bound to the request that produced it."""

    schema_version: Literal["llm-response/1.0"] = "llm-response/1.0"
    request_sha256: Digest
    request: LLMRequest
    response_text: str
    response_sha256: Digest
    provider_model_version: Text
    finish_reason: Text
    usage: TokenUsage
    # Operational measurement; excluded from any reproducible digest by consumers.
    latency_ms: Annotated[int, Field(strict=True, ge=0)]

    @model_validator(mode="after")
    def digests(self) -> Self:
        if self.request.key != self.request_sha256:
            raise ValueError("response record does not match its request digest")
        if byte_digest(self.response_text.encode()) != self.response_sha256:
            raise ValueError("response text does not match its digest")
        return self
