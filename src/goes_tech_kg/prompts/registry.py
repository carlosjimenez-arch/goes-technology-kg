"""Prompt objects whose version is the digest of their text; rendering is strict."""

import re
from dataclasses import dataclass

from goes_tech_kg.schemas.base import byte_digest

PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")


@dataclass(frozen=True)
class Prompt:
    id: str
    system: str
    user_template: str
    output_schema_version: str
    technique: str

    @property
    def version(self) -> str:
        body = self.system + "\n\x1f\n" + self.user_template
        return f"{self.id}@{byte_digest(body.encode())[:12]}"

    @property
    def variables(self) -> frozenset[str]:
        return frozenset(PLACEHOLDER.findall(self.user_template))

    def render(self, **values: str) -> str:
        missing = self.variables - values.keys()
        extra = values.keys() - self.variables
        if missing or extra:
            raise ValueError(
                f"prompt variables mismatch: missing={sorted(missing)} extra={sorted(extra)}"
            )
        rendered = self.user_template
        for name, value in values.items():
            rendered = rendered.replace("{{" + name + "}}", value)
        if PLACEHOLDER.search(rendered):
            raise ValueError("unrendered placeholder remains")
        return rendered
