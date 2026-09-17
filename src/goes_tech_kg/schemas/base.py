"""Closed immutable records and deterministic intrinsic identities."""

import hashlib
import json
import re
import unicodedata
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Grade = Annotated[StrictInt, Field(ge=2, le=6)]
Minutes = Annotated[StrictInt, Field(ge=0)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
#: Identity slugs are lowercase ASCII kebab-case so they survive any filesystem or URL.
SLUG = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, validate_default=True, revalidate_instances="always"
    )


def canonical_json(value: object) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def byte_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_id(kind: str, slug: str, intrinsic: object) -> str:
    if not SLUG.fullmatch(slug):
        raise ValueError("identity slug must be lowercase ASCII kebab-case")
    payload = {"identity_version": 1, "kind": kind, "slug": slug, "intrinsic": intrinsic}
    # Intrinsic strings are NFC; relations, contexts and editorial fields are excluded by callers.
    encoded = unicodedata.normalize("NFC", canonical_json(payload))
    return f"{kind}-{slug}-{hashlib.sha256(encoded.encode()).hexdigest()[:16]}"
