"""Explicit pipeline node: no downstream fetch before accepted admission."""

from typing import Any

from goes_tech_kg.corpus.license_gate import license_gate
from goes_tech_kg.schemas.corpus import LicenseDecision, SourceDocument


def license_gate_node(document: SourceDocument, policy: dict[str, Any]) -> LicenseDecision:
    return license_gate(document, policy)
