"""Canonical LLM request and immutable, credential-free response records."""

from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from goes_tech_kg.schemas.base import Contract, Digest, Text, digest

Provider = Literal["vertex-ai"]


class GenerationSettings(Contract):
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
        return digest(self)


class TokenUsage(Contract):
    prompt_tokens: Annotated[int, Field(strict=True, ge=0)]
    output_tokens: Annotated[int, Field(strict=True, ge=0)]
    thoughts_tokens: Annotated[int, Field(strict=True, ge=0)] = 0


class ResponseRecord(Contract):
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
        from goes_tech_kg.schemas.base import byte_digest

        if byte_digest(self.response_text.encode()) != self.response_sha256:
            raise ValueError("response text does not match its digest")
        return self
