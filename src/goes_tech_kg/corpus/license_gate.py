"""Fail-closed resource admission, independent of fetching the corpus body."""

from typing import Any
from urllib.parse import urlsplit

from goes_tech_kg.schemas.base import digest
from goes_tech_kg.schemas.corpus import LicenseDecision, SourceDocument

OPEN_LICENSES = {
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-NC-SA-4.0",
    "CC-BY-3.0-IGO",
    "CC-BY-SA-3.0-IGO",
    "OGL-3.0",
    "CC0-1.0",
}


def allowed_url(url: str, policy: dict[str, Any]) -> bool:
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.username or parts.password:
        return False
    return any(
        parts.hostname == row["host"]
        and any(parts.path.startswith(p) for p in row.get("path_prefixes", ["/"]))
        for row in policy["publishers"]
    )


def license_gate(document: SourceDocument, policy: dict[str, Any]) -> LicenseDecision:
    reason = "Verified resource-specific admission evidence"
    accepted = True
    if not allowed_url(str(document.url), policy):
        accepted = False
        reason = "URL is outside the publisher allowlist"
    elif not document.license_evidence_url or not document.license_evidence:
        accepted = False
        reason = "Missing verifiable license/publication evidence"
    elif document.admission_basis == "official_publication":
        host = urlsplit(str(document.url)).hostname
        accepted = any(
            row["host"] == host and row.get("official", False) for row in policy["publishers"]
        )
        if not accepted:
            reason = "Publisher is not registered as an official publication authority"
    elif document.admission_basis != "open_license" or document.license not in OPEN_LICENSES:
        accepted = False
        reason = "Unverified or unsupported reuse license"
    return LicenseDecision(
        document_id=document.id,
        status="accepted" if accepted else "rejected",
        reason=reason,
        policy_sha256=digest(policy),
        evidence_url=document.license_evidence_url,
        reviewed_at=document.reviewed_at,
    )
