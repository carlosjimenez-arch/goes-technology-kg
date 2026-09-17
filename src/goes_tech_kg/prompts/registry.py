"""Prompt objects whose version is the digest of their text; rendering is strict."""

import re
from dataclasses import dataclass

from goes_tech_kg.schemas.base import byte_digest

PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")


@dataclass(frozen=True)
class Prompt:
    """A versioned prompt whose version is the digest of its own text.

    Deriving the version from the text makes it impossible to edit a prompt without changing
    the replay key of every request that used it, so a recorded answer always names the exact
    wording that produced it.
    """

    id: str
    system: str
    user_template: str
    output_schema_version: str
    technique: str

    @property
    def version(self) -> str:
        """Identifier plus a digest of system text and template, stable across processes."""
        body = self.system + "\n\x1f\n" + self.user_template
        return f"{self.id}@{byte_digest(body.encode())[:12]}"

    @property
    def variables(self) -> frozenset[str]:
        """Placeholder names the template expects."""
        return frozenset(PLACEHOLDER.findall(self.user_template))

    def render(self, **values: str) -> str:
        """Fill every placeholder exactly once; a missing or surplus value is an error."""
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
