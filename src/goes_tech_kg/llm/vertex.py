"""Vertex AI Gemini adapter over REST with ADC; the only module that talks to the provider."""

import subprocess
import time
from collections.abc import Callable
from typing import Any

import httpx

from goes_tech_kg.schemas.base import byte_digest
from goes_tech_kg.schemas.llm import LLMRequest, ResponseRecord, TokenUsage

TokenProvider = Callable[[], str]


def adc_token_provider() -> TokenProvider:
    """Application Default Credentials; scoped to cloud-platform and refreshed per call."""
    import google.auth
    from google.auth.transport.requests import Request

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    def provide() -> str:
        if not credentials.valid:
            credentials.refresh(Request())  # type: ignore[no-untyped-call]
        token = credentials.token
        if not isinstance(token, str):
            raise RuntimeError("ADC did not yield an access token")
        return token

    return provide


def gcloud_token_provider() -> TokenProvider:
    """Operator fallback: the active gcloud account mints tokens when the ADC file is stale."""

    def provide() -> str:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"], capture_output=True, text=True, check=False
        )
        token = result.stdout.strip()
        if result.returncode != 0 or not token:
            raise RuntimeError("gcloud could not mint an access token: " + result.stderr[:200])
        return token

    return provide


def resilient_token_provider() -> TokenProvider:
    """ADC first; if refresh fails, fall back to the gcloud CLI and remember the choice."""
    adc = adc_token_provider()
    chosen: list[TokenProvider] = []

    def provide() -> str:
        if chosen:
            return chosen[0]()
        try:
            token = adc()
        except Exception:  # noqa: BLE001 - any refresh failure means ADC is unusable here
            fallback = gcloud_token_provider()
            token = fallback()
            chosen.append(fallback)
            return token
        chosen.append(adc)
        return token

    return provide


def endpoint(project: str, location: str, model: str) -> str:
    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    return (
        f"https://{host}/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:generateContent"
    )


def request_body(request: LLMRequest, user_content: str) -> dict[str, Any]:
    settings = request.settings
    generation: dict[str, Any] = {
        "temperature": settings.temperature,
        "seed": settings.seed,
        "maxOutputTokens": settings.max_output_tokens,
        "responseMimeType": settings.response_mime_type,
        "responseJsonSchema": request.output_json_schema,
    }
    if settings.thinking_budget is not None:
        generation["thinkingConfig"] = {"thinkingBudget": settings.thinking_budget}
    if settings.thinking_level is not None:
        generation["thinkingConfig"] = {"thinkingLevel": settings.thinking_level}
    return {
        "systemInstruction": {"parts": [{"text": request.system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
        "generationConfig": generation,
    }


class VertexClient:
    def __init__(
        self,
        project: str,
        token_provider: TokenProvider,
        client: httpx.Client | None = None,
        timeout_seconds: float = 300.0,
    ):
        self.project = project
        self.token_provider = token_provider
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def generate(self, request: LLMRequest, user_content: str) -> ResponseRecord:
        if byte_digest(user_content.encode()) != request.user_content_sha256:
            raise ValueError("user content does not match the request digest")
        headers = {"Authorization": f"Bearer {self.token_provider()}"}
        started = time.perf_counter()
        body = request_body(request, user_content)
        url = endpoint(self.project, request.location, request.model)
        try:
            response = self.client.post(url, headers=headers, json=body)
        except httpx.HTTPError as first:
            # One retry for transport faults (timeouts, resets); a second fault is recorded
            # as a transport error by the caller, never as silent output.
            try:
                response = self.client.post(url, headers=headers, json=body)
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"transport failure for {request.model}: {type(first).__name__}; retry: {exc!r}"
                ) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        if response.status_code != 200:
            raise RuntimeError(
                f"provider error {response.status_code} for {request.model}: {response.text[:300]}"
            )
        payload = response.json()
        candidates = payload.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"provider returned no candidates: {str(payload)[:300]}")
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts)
        usage = payload.get("usageMetadata", {})
        return ResponseRecord(
            request_sha256=request.key,
            request=request,
            response_text=text,
            response_sha256=byte_digest(text.encode()),
            provider_model_version=str(payload.get("modelVersion") or request.model),
            finish_reason=str(candidate.get("finishReason", "UNKNOWN")),
            usage=TokenUsage(
                prompt_tokens=int(usage.get("promptTokenCount", 0)),
                output_tokens=int(usage.get("candidatesTokenCount", 0)),
                thoughts_tokens=int(usage.get("thoughtsTokenCount", 0)),
            ),
            latency_ms=latency_ms,
        )
