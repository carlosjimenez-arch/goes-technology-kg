"""Immutable response store; replay never calls a model and never overwrites a record."""

from pathlib import Path

from goes_tech_kg.schemas.base import canonical_json
from goes_tech_kg.schemas.llm import LLMRequest, ResponseRecord


class ReplayMiss(LookupError):
    """Raised when a request has no recorded response and online acquisition is not authorized."""


class ReplayStore:
    """Content-addressed store of provider answers, keyed by the digest of their request.

    Records are immutable: an accepted answer is evidence, so a second answer to the same
    request is a conflict to resolve rather than an update to apply.
    """

    def __init__(self, root: Path):
        self.root = root

    def path(self, key: str) -> Path:
        """Where the answer to a request digest lives."""
        return self.root / f"{key}.json"

    def get(self, request: LLMRequest) -> ResponseRecord | None:
        """The recorded answer, or None; a record that does not match its request raises."""
        path = self.path(request.key)
        if not path.exists():
            return None
        record = ResponseRecord.model_validate_json(path.read_bytes())
        if record.request_sha256 != request.key or record.request != request:
            raise ValueError(f"replay record does not match request: {request.key}")
        return record

    def put(self, record: ResponseRecord) -> Path:
        """Store an answer atomically, refusing to replace a different accepted one."""
        path = self.path(record.request_sha256)
        body = (canonical_json(record) + "\n").encode()
        if path.exists():
            if path.read_bytes() != body:
                raise ValueError(
                    f"refusing to overwrite accepted response: {record.request_sha256}"
                )
            return path
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_bytes(body)
        tmp.replace(path)
        return path

    def keys(self) -> tuple[str, ...]:
        """Every recorded request digest, sorted."""
        return tuple(sorted(p.stem for p in self.root.glob("*.json")))
